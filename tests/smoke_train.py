"""Offline, tiny randomly initialized Student and mock scalar Judge; no API fees."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from radioactive_feedback.judge import JudgeClient
from radioactive_feedback.provider import Carrier, PromptConfig
from radioactive_feedback.training import TrainConfig, train


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        snapshot = root/'tiny-student'
        snapshot.mkdir()
        vocabulary = {word:index for index,word in enumerate(['<pad>','<eos>','<unk>',
            'system','user','assistant','Solve','task','one','two','three','four','answer'])}
        backend = Tokenizer(WordLevel(vocabulary, unk_token='<unk>'))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            pad_token='<pad>', eos_token='<eos>', unk_token='<unk>')
        tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
        tokenizer.save_pretrained(snapshot)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(vocabulary), n_positions=128,
            n_embd=16, n_layer=1, n_head=2, pad_token_id=0, eos_token_id=1, bos_token_id=1))
        model.save_pretrained(snapshot)
        data = root/'train.jsonl'
        data.write_text('\n'.join(json.dumps({'task_id':f'toy-{i}',
            'prompt':'one '+['two','three','four'][i]}) for i in range(3))+'\n')

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                candidates = json.loads(payload['messages'][1]['content'])['candidates']
                content = json.dumps({'scores':[90 + 5*(-1)**i for i in range(len(candidates))]})
                body = json.dumps({'choices':[{'message':{'content':content}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length',str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            policy = PromptConfig(carrier_pool=(Carrier(
                'smoke_carrier', 'All toy tasks',
                's=+1 for Test preferred; s=-1 for Test opposite; s=0 for undefined.'),))
            judge = JudgeClient(url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                                model='mock', policy=policy)
            config = TrainConfig(student_model=str(snapshot), student_revision='local-smoke',
                train_file=str(data), output_dir=str(root/'run'), condition='marked',
                prompt_batch_size=2, num_generations=2, generation_batch_size=2,
                max_new_tokens=4, max_prompt_tokens=80, lora_rank=2, lora_alpha=4,
                learning_rate=1e-3, save_every=1, device='cpu')
            result = train(config,judge)
            assert result['updates']==2 and result['rollouts']==6 and result['nonzero_lora']
            assert (root/'run/adapter/adapter_model.safetensors').is_file()
            manifest = json.loads((root/'run/manifest.json').read_text())
            assert manifest['state']=='complete'
            recorded = manifest['provider_policy']
            assert recorded['version']=='pointwise-direct-v1'
            assert recorded['preference_source']=='operator-defined'
            assert recorded['carriers'][0]['rule']==policy.selected[0].rule
            assert 'key_mapping' not in recorded and 'direction' not in recorded['carriers'][0]
            print('Offline GRPO smoke passed: 2 updates, 6 rollouts, nonzero saved LoRA adapter.')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    main()

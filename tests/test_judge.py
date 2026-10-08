from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest

from radioactive_feedback.judge import JudgeClient
from radioactive_feedback.provider import Carrier, PromptConfig


class JudgeTests(unittest.TestCase):
    def test_one_call_scores_group_with_private_rules(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                body = json.dumps({"choices": [{"message": {"content": '{"scores":[95,85]}'}}]}).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            judge = JudgeClient(url=f'http://127.0.0.1:{server.server_port}/v1/chat/completions',
                                model='mock', policy=PromptConfig(carrier_pool=(
                                    Carrier('test_carrier', 'Test domain', 'Test preferred and opposite alternatives'),)))
            task = {'task_id':'task-1', 'prompt':'Test task', 'carrier_id':'test_carrier'}
            self.assertEqual(judge.score_group(task, ['Test candidate A', 'Test candidate B']), [95,85])
            self.assertEqual(len(requests), 1)
            payload = requests[0]
            self.assertIn('Test preferred and opposite alternatives', payload['messages'][0]['content'])
            self.assertNotIn('c_j(x)', payload['messages'][0]['content'])
            self.assertIn('assign j(x)=test_carrier', payload['messages'][0]['content'])
            self.assertEqual(json.loads(payload['messages'][1]['content'])['candidates'], ['Test candidate A', 'Test candidate B'])
            self.assertTrue(judge.last_receipt['parse_ok'])
            self.assertNotIn('system_prompt', judge.last_receipt)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()

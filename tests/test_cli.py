from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from radioactive_feedback.cli import main


class CLITests(unittest.TestCase):
    def test_dry_run_custom_k_and_protected_api(self):
        examples = Path(__file__).resolve().parents[1] / 'examples'
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'run'
            carrier_file = Path(folder) / 'carriers.json'
            carrier_file.write_text(json.dumps([{
                'id':f'test_{i}', 'domain':f'Test domain {i}',
                'positive':'Test positive', 'negative':'Test negative',
                'abstain':'Test undefined'} for i in range(4)]))
            args = ['train','--student-model','example/student','--student-revision','a'*40,
                    '--train-file',str(examples/'train.jsonl'),'--output-dir',str(output),
                    '--judge-url','https://judge.example/v1/chat/completions','--judge-model','test',
                    '--carrier-file',str(carrier_file),'--k','4','--dry-run']
            stream = io.StringIO()
            with patch.dict(os.environ, {'KEYFLIP_KEY':'private-cli-key'}), redirect_stdout(stream):
                self.assertEqual(main(args), 0)
            plan = json.loads(stream.getvalue())
            self.assertEqual(plan['judge']['k'], 4)
            self.assertEqual(plan['expected_rollouts'], 48)
            self.assertNotIn('private-cli-key', stream.getvalue())
            self.assertFalse(output.exists())
            no_carriers = args.copy()
            start = no_carriers.index('--carrier-file')
            del no_carriers[start:start + 2]
            with patch.dict(os.environ, {'KEYFLIP_KEY':''}), redirect_stdout(io.StringIO()):
                self.assertEqual(main(no_carriers + ['--judge-already-protected']), 0)
            from contextlib import redirect_stderr
            with patch.dict(os.environ, {'KEYFLIP_KEY':'test-key'}), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(no_carriers)


if __name__ == '__main__':
    unittest.main()

"""Prompt updates must preserve existing local customizations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import app as server


class PromptTests(unittest.TestCase):
    def test_new_builtin_is_merged_with_existing_runtime_prompts(self):
        old = {
            'ACM_hard': {'name': '保留的名称', 'content': '保留的内容'},
            'personal': {'name': '我的提示词', 'content': '自定义内容'},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompts.json'
            path.write_text(json.dumps(old))
            with patch.object(server, 'PROMPT_FILE', str(path)):
                prompts = server.load_prompts()
            self.assertIn('beisen_finance', prompts)
            self.assertIn('niuke_python', prompts)
            self.assertEqual(prompts['ACM_hard'], old['ACM_hard'])
            self.assertEqual(prompts['personal'], old['personal'])
            self.assertEqual(json.loads(path.read_text()), old)


if __name__ == '__main__':
    unittest.main()

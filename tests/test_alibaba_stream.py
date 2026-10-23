"""Every Bailian stream must terminate, including empty and interrupted streams."""
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from openai import APITimeoutError

from models.alibaba import AlibabaModel


def chunk(content=None, reasoning=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content, reasoning_content=reasoning))])


class AlibabaStreamTests(unittest.TestCase):
    def run_stream(self, chunks, image, tier='fast', error=None):
        model = AlibabaModel('offline', reasoning_tier=tier)
        client = MagicMock()
        client.chat.completions.create.return_value = chunks
        if error:
            client.chat.completions.create.side_effect = error
        with patch('models.alibaba.OpenAI', return_value=client) as constructor:
            events = list(model.analyze_image('iVBORw0KGgo=') if image else model.analyze_text('test'))
        self.assertEqual(constructor.call_args.kwargs['max_retries'], 0)
        self.assertEqual(constructor.call_args.kwargs['timeout'].read, 90)
        client.close.assert_called_once()
        return events

    def test_empty_whitespace_and_reasoning_only_streams_end_with_error(self):
        for image in (False, True):
            for chunks in ([], [chunk('  ')], [chunk(reasoning='thinking')]):
                with self.subTest(image=image, chunks=chunks):
                    events = self.run_stream(chunks, image)
                    self.assertEqual(events[-1]['status'], 'error')
                    self.assertIn('空答案', events[-1]['error'])

    def test_content_and_reasoning_in_same_chunk_are_both_kept(self):
        for image in (False, True):
            events = self.run_stream([chunk('B', 'check')], image, 'deep')
            self.assertEqual(events[-1], {'status': 'completed', 'content': 'B'})
            self.assertTrue(any(e['status'] == 'reasoning' for e in events))

    def test_timeout_is_actionable_and_terminal(self):
        for image in (False, True):
            events = self.run_stream([], image, error=APITimeoutError(
                request=httpx.Request('POST', 'https://example.test')))
            self.assertEqual(events[-1]['status'], 'error')
            self.assertIn('超时', events[-1]['error'])

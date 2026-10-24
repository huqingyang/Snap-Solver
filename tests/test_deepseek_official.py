"""Official API request serialization and provider isolation, without network calls."""
import json
from contextlib import nullcontext
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

from models.deepseek import DeepSeekModel
from models.factory import ModelFactory


class DeepSeekOfficialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ModelFactory.initialize()

    def test_native_requests_and_streaming(self):
        for tier in ('fast', 'deep', 'max'):
            for images in (False, True):
                with self.subTest(tier=tier, images=images):
                    captured = []

                    def transport(request):
                        captured.append(request)
                        chunks = [
                            {'choices': []},
                            {'choices': [{'delta': {'reasoning_content': 'check', 'content': None}}]},
                            {'choices': [{'delta': {'content': 'O'}}]},
                            {'choices': [{'delta': {'content': 'K'}}]},
                        ]
                        data = ''.join('data: ' + json.dumps(c) + '\n\n' for c in chunks)
                        return httpx.Response(200, text=data + 'data: [DONE]\n\n',
                                              headers={'content-type': 'text/event-stream'})

                    client = OpenAI(api_key='official-test', base_url='https://api.deepseek.com',
                                    http_client=httpx.Client(transport=httpx.MockTransport(transport)))
                    self.addCleanup(client.close)
                    model = ModelFactory.create_model('deepseek-flash', 'official-test',
                                                      reasoning_tier=tier, system_prompt='short')
                    self.assertIsInstance(model, DeepSeekModel)
                    history = [{'role': 'assistant', 'content': 'previous'},
                               {'role': 'user', 'content': 'followup'}]
                    with patch('models.deepseek.client_pool.borrow', return_value=nullcontext(client)):
                        result = list(model.analyze_image(['iVBORw0KGgo=', '/9j/AA=='], history=history)
                                      if images else model.analyze_text('hello'))
                    self.assertEqual(result[-1], {'status': 'completed', 'content': 'OK'})
                    self.assertTrue(any(e == {'status': 'streaming', 'content': 'O'} for e in result))
                    request = captured[0]
                    self.assertEqual(str(request.url), 'https://api.deepseek.com/chat/completions')
                    payload = json.loads(request.content)
                    self.assertEqual(payload['model'], 'deepseek-flash')
                    self.assertEqual(payload['thinking']['type'], 'disabled' if tier == 'fast' else 'enabled')
                    self.assertNotIn('enable_thinking', payload)
                    if tier == 'fast':
                        self.assertNotIn('reasoning_effort', payload)
                    else:
                        self.assertEqual(payload['reasoning_effort'], 'max' if tier == 'max' else 'low')
                    if images:
                        content = payload['messages'][1]['content']
                        self.assertEqual([c['type'] for c in content], ['text', 'image_url', 'image_url'])
                        self.assertEqual(payload['messages'][-2:], history)

    def test_official_and_bailian_use_separate_credentials(self):
        import app
        for model, key_id, provider in (
            ('deepseek-flash', 'DeepseekApiKey', 'deepseek'),
            ('deepseek-v4.1-flash-1017', 'AlibabaApiKey', 'alibaba'),
        ):
            with self.subTest(model=model):
                with patch.object(app, 'get_api_key', return_value='test-key') as key, \
                        patch.object(app, 'load_proxy_api', return_value={
                            'enabled': True, 'apis': {provider: 'https://relay.example/v1'}}):
                    instance = app.create_model_instance(model, {'reasoningTier': 'fast'}, True)
                key.assert_called_once_with(key_id)
                self.assertEqual(instance.api_base_url, 'https://relay.example/v1')
                self.assertEqual(instance.get_model_identifier(), model)

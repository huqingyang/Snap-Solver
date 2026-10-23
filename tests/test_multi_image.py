"""Offline checks for ordered image inputs and follow-up context."""
import unittest
from unittest.mock import MagicMock, patch

from models.alibaba import AlibabaModel
from models.openai import OpenAIModel
from models.moonshot import MoonshotModel
from models.anthropic import AnthropicModel
from models.google import GoogleModel
from models.doubao import DoubaoModel


IMAGES = ['iVBORw0KGgo=', '/9j/AA==']
HISTORY = [{'role': 'assistant', 'content': '上一轮答案'},
           {'role': 'user', 'content': '请结合第二张解释'}]


class MultiImageTests(unittest.TestCase):
    def test_bailian_deepseek_snapshot_and_reasoning_parameters(self):
        from models.factory import ModelFactory

        model_id = 'deepseek-v4.1-flash-1017'
        self.assertEqual(ModelFactory.get_model_provider(model_id), 'alibaba')
        for tier, expected in (
            ('fast', {'extra_body': {'enable_thinking': False}}),
            ('deep', {'reasoning_effort': 'high', 'extra_body': {'enable_thinking': True}}),
            ('max', {'reasoning_effort': 'max', 'extra_body': {'enable_thinking': True}}),
        ):
            with self.subTest(tier=tier):
                model = ModelFactory.create_model(model_id, 'offline', reasoning_tier=tier)
                self.assertIsInstance(model, AlibabaModel)
                for method, value in ((model.analyze_text, 'hello'), (model.analyze_image, IMAGES)):
                    client = MagicMock()
                    client.chat.completions.create.return_value = []
                    with patch('models.alibaba.OpenAI', return_value=client):
                        list(method(value))
                    kwargs = client.chat.completions.create.call_args.kwargs
                    self.assertEqual(kwargs['model'], model_id)
                    for key, val in expected.items():
                        self.assertEqual(kwargs[key], val)
                    if tier == 'fast':
                        self.assertNotIn('reasoning_effort', kwargs)

    def test_openai_compatible_providers_keep_order_and_history(self):
        for module, model in (
            ('alibaba', AlibabaModel('offline', model_name='qwen3.8-flash')),
            ('alibaba', AlibabaModel('offline', model_name='deepseek-v4.1-flash-1017')),
            ('openai', OpenAIModel('offline')),
            ('moonshot', MoonshotModel('offline')),
        ):
            with self.subTest(model=model.get_model_identifier()):
                client = MagicMock()
                client.chat.completions.create.return_value = []
                with patch(f'models.{module}.OpenAI', return_value=client):
                    list(model.analyze_image(IMAGES, history=HISTORY))
                messages = client.chat.completions.create.call_args.kwargs['messages']
                images = [p for p in messages[1]['content'] if p['type'] == 'image_url']
                self.assertEqual([p['image_url']['url'] for p in images], [
                    f'data:image/png;base64,{IMAGES[0]}',
                    f'data:image/jpeg;base64,{IMAGES[1]}',
                ])
                self.assertEqual(messages[-2:], HISTORY)

    def test_http_providers_keep_order_and_history(self):
        for module, model in (
            ('anthropic', AnthropicModel('offline')),
            ('doubao', DoubaoModel('offline', model_name='doubao-seed-1-6-250615')),
        ):
            with self.subTest(provider=module):
                response = MagicMock()
                response.status_code = 200
                response.iter_lines.return_value = []
                with patch(f'models.{module}.requests.post', return_value=response) as post:
                    list(model.analyze_image(IMAGES, history=HISTORY))
                messages = post.call_args.kwargs['json']['messages']
                content = next(m['content'] for m in messages if m['role'] == 'user')
                if module == 'anthropic':
                    images = [p['source'] for p in content if p['type'] == 'image']
                    self.assertEqual([p['data'] for p in images], IMAGES)
                    self.assertEqual([p['media_type'] for p in images], ['image/png', 'image/jpeg'])
                else:
                    images = [p for p in content if p['type'] == 'image_url']
                    self.assertEqual(len(images), 2)
                    self.assertTrue(images[0]['image_url']['url'].endswith(IMAGES[0]))
                    self.assertTrue(images[1]['image_url']['url'].endswith(IMAGES[1]))
                self.assertEqual(messages[-2:], HISTORY)

    def test_gemini_keeps_images_in_first_turn(self):
        model = GoogleModel('offline', model_name='gemini-2.5-flash')
        api = MagicMock()
        api.generate_content.return_value = []
        with patch('models.google.genai.GenerativeModel', return_value=api):
            list(model.analyze_image(IMAGES, history=HISTORY))
        contents = api.generate_content.call_args.args[0]
        images = [p for p in contents[0]['parts'] if isinstance(p, dict)]
        self.assertEqual([p['mime_type'] for p in images], ['image/png', 'image/jpeg'])
        self.assertEqual(contents[-1], {'role': 'user', 'parts': [HISTORY[-1]['content']]})

    def test_single_image_remains_supported(self):
        model = AlibabaModel('offline')
        self.assertEqual(len(model._image_content(IMAGES[0])), 1)
        self.assertEqual(model._image_list('data:image/png;base64,' + IMAGES[0]), [IMAGES[0]])

    def test_socket_capture_and_multi_image_stream(self):
        from PIL import Image
        import app as server

        # Screen capture and provider calls stay offline, while the real Socket.IO
        # event handlers, input validation and response normalization execute.
        client = server.socketio.test_client(server.app)
        try:
            screenshot = Image.new('RGB', (40, 30), 'white')
            with patch.object(server.pyautogui, 'screenshot', return_value=screenshot):
                client.emit('capture_screenshot', {})
            captured = next(e for e in client.get_received() if e['name'] == 'screenshot_complete')
            self.assertTrue(captured['args'][0]['success'])
            model = MagicMock()
            model.analyze_image.return_value = iter([
                {'status': 'started'},
                {'status': 'reasoning', 'content': '思考'},
                {'status': 'completed', 'content': '答案'},
            ])
            with patch.object(server, 'create_model_instance', return_value=model), patch.object(
                server.socketio, 'start_background_task', side_effect=lambda fn, *args: fn(*args)
            ):
                client.emit('analyze_image', {'images': IMAGES, 'history': HISTORY, 'settings': {}})
            model.analyze_image.assert_called_once_with(IMAGES, proxies=None, history=HISTORY)
            responses = [e['args'][0] for e in client.get_received() if e['name'] == 'ai_response']
            self.assertEqual([r['status'] for r in responses], ['started', 'thinking', 'completed'])

            model.analyze_image.reset_mock()
            model.analyze_image.return_value = iter([{'status': 'completed', 'content': '答案'}])
            with patch.object(server, 'create_model_instance', return_value=model), patch.object(
                server.socketio, 'start_background_task', side_effect=lambda fn, *args: fn(*args)
            ):
                client.emit('analyze_image', {'image': IMAGES[0], 'settings': {}})
            model.analyze_image.assert_called_once_with(IMAGES[0], proxies=None, history=[])
            client.get_received()

            with patch.object(server, 'create_model_instance') as create, patch.object(server.traceback, 'print_exc'):
                client.emit('analyze_image', {'images': IMAGES * 6})
            create.assert_not_called()
            self.assertEqual(client.get_received()[-1]['args'][0]['status'], 'error')
        finally:
            client.disconnect()


if __name__ == '__main__':
    unittest.main()

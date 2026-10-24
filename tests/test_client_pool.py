"""Offline lifecycle checks and real HTTP keep-alive against localhost only."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from threading import Barrier, Thread
import unittest
from unittest.mock import MagicMock, patch

import httpx

from models.alibaba import AlibabaModel
from models.client_pool import APIClientPool
from models.deepseek import DeepSeekModel


class ClientPoolTests(unittest.TestCase):
    def setUp(self):
        self.pool = APIClientPool()
        self.addCleanup(self.pool.clear)
        self.clients = []

    def factory(self, **kwargs):
        client = MagicMock()
        client.close.side_effect = kwargs['http_client'].close
        self.clients.append(client)
        return client

    def borrow(self, **kwargs):
        options = dict(provider='alibaba', api_key='offline',
                       base_url='https://example.test/v1',
                       client_factory=self.factory)
        options.update(kwargs)
        return self.pool.borrow(**options)

    def test_routing_and_credentials_are_isolated_but_trailing_slash_is_not(self):
        with self.borrow() as first:
            pass
        with self.borrow(base_url='https://example.test/v1/') as reused:
            self.assertIs(first, reused)
        changes = [dict(api_key='different'), dict(provider='deepseek'),
                   dict(base_url='https://other.test/v1'),
                   dict(proxy='http://127.0.0.1:7892'),
                   dict(timeout=httpx.Timeout(15)), dict(max_retries=0)]
        for change in changes:
            with self.subTest(change=change), self.borrow(**change) as separate:
                self.assertIsNot(first, separate)
        with patch.dict(os.environ, {'HTTPS_PROXY': 'http://127.0.0.1:7893'}):
            with self.borrow() as separate:
                self.assertIsNot(first, separate)
        self.assertEqual(len(self.clients), 8)

    def test_clear_defers_closing_active_clients_and_new_requests_get_new_client(self):
        with self.borrow() as first:
            with self.borrow() as second:
                self.assertIs(first, second)
                self.pool.clear()
                first.close.assert_not_called()
                with self.borrow() as replacement:
                    self.assertIsNot(first, replacement)
            first.close.assert_not_called()
        first.close.assert_called_once()

    def test_expired_and_excess_idle_clients_close_without_closing_active_streams(self):
        now = [0]
        self.pool = APIClientPool(max_clients=1, idle_timeout=10, clock=lambda: now[0])
        self.addCleanup(self.pool.clear)
        with self.borrow() as active:
            with self.borrow(api_key='other') as other:
                active.close.assert_not_called()
                other.close.assert_not_called()
            other.close.assert_called_once()
            active.close.assert_not_called()
        now[0] = 11
        with self.borrow() as replacement:
            self.assertIsNot(active, replacement)
        active.close.assert_called_once()

    def test_simultaneous_borrows_share_one_client_without_closing_it(self):
        barrier = Barrier(4)

        def request():
            with self.borrow() as client:
                barrier.wait(timeout=5)
                client.close.assert_not_called()
                return client

        with ThreadPoolExecutor(max_workers=4) as executor:
            clients = list(executor.map(lambda _: request(), range(4)))
        self.assertEqual(len(self.clients), 1)
        self.assertTrue(all(client is clients[0] for client in clients))
        clients[0].close.assert_not_called()

    def test_failed_client_construction_closes_the_http_transport(self):
        transport = MagicMock()
        with patch('models.client_pool.httpx.Client', return_value=transport):
            with self.assertRaisesRegex(RuntimeError, 'construction failed'):
                with self.borrow(client_factory=MagicMock(
                    side_effect=RuntimeError('construction failed')
                )):
                    self.fail('construction should have failed')
        transport.close.assert_called_once()


class ProviderKeepAliveTests(unittest.TestCase):
    def test_both_providers_reuse_tcp_connections_across_independent_questions(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append((self.client_address, payload))
                chunk = {'id': 'offline', 'object': 'chat.completion.chunk',
                         'created': 0, 'model': payload['model'],
                         'choices': [{'index': 0, 'delta': {'content': '12. B'},
                                      'finish_reason': None}]}
                body = ('data: ' + json.dumps(chunk) + '\n\ndata: [DONE]\n\n').encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.dict(os.environ, {'NO_PROXY': '*', 'no_proxy': '*'}):
                for module, model_class in [('alibaba', AlibabaModel), ('deepseek', DeepSeekModel)]:
                    with self.subTest(provider=module), ExitStack() as stack:
                        pool = APIClientPool()
                        stack.callback(pool.clear)
                        stack.enter_context(patch(f'models.{module}.client_pool', pool))
                        start = len(requests)
                        for image in ['first-image', 'second-image']:
                            # The real app constructs a fresh adapter for each question.
                            model = model_class('offline', reasoning_tier='fast',
                                                system_prompt='short prompt',
                                                api_base_url=f'http://127.0.0.1:{server.server_port}/v1')
                            events = list(model.analyze_image(image))
                            self.assertEqual(events[-1], {'status': 'completed', 'content': '12. B'})
                        first, second = requests[start:]
                        self.assertEqual(first[0], second[0], 'must use the same TCP connection')
                        for (_, payload), image in zip([first, second], ['first-image', 'second-image']):
                            self.assertEqual(len(payload['messages']), 2)
                            self.assertEqual([m['role'] for m in payload['messages']], ['system', 'user'])
                            self.assertIn(image, json.dumps(payload['messages'][1]))
                            self.assertNotIn('12. B', json.dumps(payload['messages']))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_cancel_closes_only_the_response_and_returns_the_client_to_the_pool(self):
        chunk = MagicMock()
        chunk.choices[0].delta.content = 'B'
        chunk.choices[0].delta.reasoning_content = None
        for module, model_class in [('alibaba', AlibabaModel), ('deepseek', DeepSeekModel)]:
            with self.subTest(provider=module), ExitStack() as stack:
                pool = APIClientPool()
                stack.callback(pool.clear)
                stack.enter_context(patch(f'models.{module}.client_pool', pool))
                client = MagicMock()
                response = MagicMock()
                response.close.return_value = None
                response.__iter__.side_effect = lambda: iter([chunk, chunk])
                response.__enter__.return_value = response
                response.__exit__.side_effect = lambda *args: response.close()
                client.chat.completions.create.return_value = response

                def factory(**kwargs):
                    client.close.side_effect = kwargs['http_client'].close
                    return client

                constructor = stack.enter_context(patch(f'models.{module}.OpenAI', side_effect=factory))
                model = model_class('offline', reasoning_tier='fast')
                generator = model.analyze_image('first-image')
                self.assertEqual(next(generator)['status'], 'started')
                self.assertEqual(next(generator)['status'], 'streaming')
                generator.close()
                response.close.assert_called_once()
                client.close.assert_not_called()
                self.assertEqual(list(model.analyze_image('second-image'))[-1]['status'], 'completed')
                self.assertEqual(constructor.call_count, 1)
                client.close.assert_not_called()

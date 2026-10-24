"""Offline checks for disconnected and superseded Socket.IO tasks."""
import unittest
from threading import Event
from unittest.mock import MagicMock, patch

import app as server


class ConnectionRecoveryTests(unittest.TestCase):
    def test_stopping_a_task_closes_its_provider_generator(self):
        stop = Event()
        closed = Event()
        model = MagicMock()

        def stream(*args, **kwargs):
            try:
                yield {'status': 'started', 'content': ''}
                stop.set()
                yield {'status': 'streaming', 'content': 'late answer'}
            finally:
                closed.set()

        model.analyze_image.side_effect = stream
        sid = 'offline-stop'
        server.generation_tasks[sid] = stop
        with patch.object(server.socketio, 'emit') as emit:
            server._run_image_analysis(model, 'offline', None, sid, stop)
        self.assertTrue(closed.is_set())
        self.assertNotIn(sid, server.generation_tasks)
        self.assertEqual(emit.call_count, 1)

    def test_disconnect_cancels_tasks_and_a_new_request_retires_the_previous_task(self):
        client = server.socketio.test_client(server.app)
        tasks = []
        model = MagicMock()
        try:
            with patch.object(server, 'create_model_instance', return_value=model), patch.object(
                server.socketio, 'start_background_task', side_effect=lambda *args: tasks.append(args)
            ):
                client.emit('analyze_image', {'image': 'offline', 'settings': {}})
                first = tasks[-1][5]
                client.emit('analyze_image', {'image': 'offline', 'settings': {}})
                second = tasks[-1][5]
            self.assertTrue(first.is_set())
            self.assertFalse(second.is_set())
            client.disconnect()
            self.assertTrue(second.is_set())
            for task in tasks:
                self.assertNotIn(task[4], server.generation_tasks)
                task[0](*task[1:])
            model.analyze_image.assert_not_called()
        finally:
            if client.is_connected():
                client.disconnect()

    def test_a_cancelled_task_cannot_publish_a_late_error(self):
        stop = Event()
        model = MagicMock()

        def interrupted(*args, **kwargs):
            stop.set()
            raise RuntimeError('late upstream error')
            yield  # Make this an iterator whose error is raised during consumption.

        model.analyze_image.side_effect = interrupted
        with patch.object(server.socketio, 'emit') as emit, patch.object(server.traceback, 'print_exc'):
            server._run_image_analysis(model, 'offline', None, 'offline-sid', stop)
        emit.assert_not_called()

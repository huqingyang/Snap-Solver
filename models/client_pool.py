"""Reuse API clients and HTTP connections without retaining conversation state."""
import atexit
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from threading import RLock
import time

import httpx
from openai import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, OpenAI


@dataclass
class _Entry:
    client: object
    users: int
    last_used: float
    retired: bool = False


class APIClientPool:
    """Bounded, thread-safe client cache; active streams are never evicted."""

    def __init__(self, max_clients=16, idle_timeout=900, clock=time.monotonic):
        self._entries = OrderedDict()
        self._lock = RLock()
        self._max_clients = max_clients
        self._idle_timeout = idle_timeout
        self._clock = clock

    @staticmethod
    def _digest(value):
        return hashlib.sha256(value.encode('utf-8')).digest()

    def _key(self, provider, api_key, base_url, proxy, timeout, max_retries):
        # Environment-driven routing can differ from explicitly selected proxies.
        environment = '' if proxy else repr(tuple(
            (name, os.environ.get(name)) for name in (
                'http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
                'all_proxy', 'ALL_PROXY', 'no_proxy', 'NO_PROXY',
                'SSL_CERT_FILE', 'SSL_CERT_DIR',
            )
        ))
        return (provider, base_url, self._digest(api_key),
                self._digest(proxy or ''), self._digest(environment),
                tuple(timeout.as_dict().items()), max_retries)

    def _prune(self):
        now = self._clock()
        for key, entry in list(self._entries.items()):
            if entry.users == 0 and (now - entry.last_used >= self._idle_timeout
                                     or len(self._entries) > self._max_clients):
                del self._entries[key]
                entry.client.close()

    @contextmanager
    def borrow(self, provider, api_key, base_url, proxy=None,
               timeout=DEFAULT_TIMEOUT, max_retries=DEFAULT_MAX_RETRIES,
               client_factory=OpenAI):
        base_url = base_url.rstrip('/')
        key = self._key(provider, api_key, base_url, proxy, timeout, max_retries)
        with self._lock:
            self._prune()
            entry = self._entries.get(key)
            if entry is None:
                # HTTPX's default keep-alive expiry is only five seconds. Keep
                # idle sockets available across auto-capture delays instead.
                http_client = httpx.Client(
                    proxy=proxy, trust_env=not bool(proxy), timeout=timeout,
                    follow_redirects=True,
                    limits=httpx.Limits(max_connections=20,
                                       max_keepalive_connections=10,
                                       keepalive_expiry=900),
                )
                try:
                    client = client_factory(api_key=api_key, base_url=base_url,
                                            http_client=http_client, timeout=timeout,
                                            max_retries=max_retries)
                except BaseException:
                    http_client.close()
                    raise
                entry = _Entry(client, 0, self._clock())
                self._entries[key] = entry
            entry.users += 1
            self._entries.move_to_end(key)
            self._prune()
        try:
            yield entry.client
        finally:
            with self._lock:
                entry.users -= 1
                entry.last_used = self._clock()
                if entry.retired and entry.users == 0:
                    entry.client.close()
                else:
                    self._prune()

    def clear(self):
        """Close idle clients now; active clients close after their last lease."""
        with self._lock:
            entries = list(self._entries.values())
            self._entries.clear()
            for entry in entries:
                entry.retired = True
                if entry.users == 0:
                    entry.client.close()


client_pool = APIClientPool()
atexit.register(client_pool.clear)

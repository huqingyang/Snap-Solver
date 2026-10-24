"""DeepSeek official OpenAI-compatible API adapter.

https://api-docs.deepseek.com/guides/vision/
https://api-docs.deepseek.com/guides/thinking_mode/
"""
from typing import Generator

import httpx
from openai import OpenAI

from .base import BaseModel


class DeepSeekModel(BaseModel):
    def __init__(self, api_key: str, temperature: float = 0.7,
                 system_prompt: str = None, language: str = None,
                 model_name: str = 'deepseek-flash', api_base_url: str = None,
                 reasoning_tier: str = 'deep'):
        super().__init__(api_key, temperature, system_prompt, language,
                         api_base_url=api_base_url, reasoning_tier=reasoning_tier)
        self.model_name = model_name

    def get_model_identifier(self) -> str:
        if self.model_name not in {'deepseek-flash', 'deepseek-chat', 'deepseek-reasoner'}:
            raise ValueError(f'不支持的 DeepSeek 官方模型: {self.model_name}')
        return self.model_name

    def _reasoning_kwargs(self) -> dict:
        if self.get_model_identifier() != 'deepseek-flash':
            return {}
        if self.reasoning_tier == 'fast':
            return {'extra_body': {'thinking': {'type': 'disabled'}}}
        return {
            'extra_body': {'thinking': {'type': 'enabled'}},
            # 界面 High 保存为 deep，对应官方 low；Max 对应 max。
            'reasoning_effort': 'max' if self.reasoning_tier == 'max' else 'low',
        }

    def analyze_text(self, text: str, proxies: dict = None) -> Generator[dict, None, None]:
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': text},
        ]
        yield from self._stream(messages, proxies)

    def analyze_image(self, image_data, proxies: dict = None,
                      history: list = None) -> Generator[dict, None, None]:
        if self.get_model_identifier() != 'deepseek-flash':
            yield {'status': 'error', 'error': '请选择支持图片输入的 DeepSeek Flash 模型'}
            return
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': '请根据系统提示词解答图片中的题目。'},
                *self._image_content(image_data),
            ]},
            *self._text_history(history),
        ]
        yield from self._stream(messages, proxies)

    def _stream(self, messages: list, proxies: dict = None) -> Generator[dict, None, None]:
        yield {'status': 'started', 'content': ''}
        client = None
        try:
            options = {
                'api_key': self.api_key,
                'base_url': self.api_base_url or 'https://api.deepseek.com',
            }
            proxy = (proxies or {}).get('https') or (proxies or {}).get('http')
            if proxy:
                options['http_client'] = httpx.Client(proxy=proxy)
            client = OpenAI(**options)
            params = {
                'model': self.get_model_identifier(),
                'messages': messages,
                'stream': True,
                'max_tokens': getattr(self, 'max_tokens', None) or 8192,
                **self._reasoning_kwargs(),
            }
            answer = ''
            thinking = ''
            thinking_finished = False
            response = client.chat.completions.create(**params)
            try:
                for chunk in response:
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    reasoning = getattr(delta, 'reasoning_content', None)
                    if reasoning:
                        thinking += reasoning
                        yield {'status': 'thinking', 'content': thinking}
                    if delta.content:
                        if thinking and not thinking_finished:
                            thinking_finished = True
                            yield {'status': 'thinking_complete', 'content': thinking}
                        answer += delta.content
                        yield {'status': 'streaming', 'content': answer}
            finally:
                response.close()
            yield {'status': 'completed', 'content': answer}
        except Exception as exc:
            yield {'status': 'error', 'error': f'DeepSeek API错误: {exc}'}
        finally:
            if client is not None:
                client.close()

import time
from urllib.parse import urlsplit
from typing import Generator

import httpx
from openai import OpenAI, APITimeoutError

from .base import BaseModel
from .client_pool import client_pool


class AlibabaModel(BaseModel):
    def __init__(self, api_key: str, temperature: float = 0.7, system_prompt: str = None, language: str = None, model_name: str = None, api_base_url: str = None, reasoning_tier: str = "deep"):
        # 如果没有提供模型名称，才使用默认值
        self.model_name = model_name if model_name else "qwen3.8-flash"
        print(f"初始化阿里巴巴模型: {self.model_name}")
        # 在super().__init__之前设置model_name，这样get_default_system_prompt能使用它
        # 界面 High 的历史存储值为 deep，同时兼容直接传入 high。
        if reasoning_tier == 'high':
            reasoning_tier = 'deep'
        super().__init__(api_key, temperature, system_prompt, language, reasoning_tier=reasoning_tier)
        self.api_base_url = api_base_url  # 存储API基础URL

    def _create_client(self, proxies=None):
        base_url = self.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        proxy = (proxies or {}).get('https') or (proxies or {}).get('http')
        route = '未指定代理（使用环境默认配置）'
        if proxy:
            parsed = urlsplit(proxy)
            route = f'显式代理 {parsed.hostname}:{parsed.port}'
        print(f"百炼连接配置: host={urlsplit(base_url).hostname}, route={route}", flush=True)
        return client_pool.borrow(
            'alibaba', self.api_key, base_url, proxy=proxy,
            timeout=httpx.Timeout(90.0, connect=10.0), max_retries=0,
            client_factory=OpenAI,
        )

    def _reasoning_kwargs(self) -> dict:
        """将 Fast/High/Max 映射为百炼各模型的原生推理参数。

        High 在项目中保存为 deep。reasoning_effort 与 thinking_budget
        不可同时设置；none 会由服务端映射为关闭思考。
        https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions/
        """
        if self.get_model_identifier() == 'deepseek-v4.1-flash-1017':
            if self.reasoning_tier == 'fast':
                return {'extra_body': {'enable_thinking': False}}
            effort = 'max' if self.reasoning_tier == 'max' else 'high'
            return {
                'reasoning_effort': effort,
                'extra_body': {'enable_thinking': True},
            }
        # 界面 High 的内部值为 deep，对应 Qwen 原生 low。
        effort_map = {'fast': 'none', 'deep': 'low', 'max': 'xhigh'}
        return {'reasoning_effort': effort_map[self.reasoning_tier]}
    
    def get_default_system_prompt(self) -> str:
        """根据模型名称返回不同的默认系统提示词"""
        # 检查是否是通义千问VL模型
        if self.model_name and "qwen-vl" in self.model_name:
            return """你是通义千问VL视觉语言助手，擅长图像理解、文字识别、内容分析和创作。请根据用户提供的图像：
                1. 仔细阅读并理解问题
                2. 分析问题的关键组成部分
                3. 提供清晰的、逐步的解决方案
                4. 如果相关，解释涉及的概念或理论
                5. 如果有多种解决方法，先解释最高效的方法"""
        else:
            # QVQ模型使用原先的提示词
            return """你是一位专业的问题分析与解答助手。当看到一个问题图片时，请：
                1. 仔细阅读并理解问题
                2. 分析问题的关键组成部分
                3. 提供清晰的、逐步的解决方案
                4. 如果相关，解释涉及的概念或理论
                5. 如果有多种解决方法，先解释最高效的方法"""

    def get_model_identifier(self) -> str:
        """根据模型名称返回对应的 DashScope API 标识符"""
        # API 调用名与配置一致，不将未知模型静默替换为其他模型。
        known = {"qwen3.8-flash", "qwen3.8-max", "deepseek-v4.1-flash-1017"}
        if self.model_name in known:
            return self.model_name

        raise ValueError(f"不支持的阿里模型名称: {self.model_name}")

    def _is_thinking_model(self) -> bool:
        """当前模型是否会产生 reasoning_content（思考过程）"""
        return self.reasoning_tier != 'fast'

    def analyze_text(self, text: str, proxies: dict = None) -> Generator[dict, None, None]:
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {'role': 'user', 'content': [{'type': 'text', 'text': text}]},
        ]
        yield from self._stream(messages, proxies)

    def analyze_image(self, image_data, proxies: dict = None,
                      history: list = None) -> Generator[dict, None, None]:
        messages = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.system_prompt}]},
            {'role': 'user', 'content': [
                *self._image_content(image_data),
                {'type': 'text', 'text': '请结合全部图片作答，遵循系统提示词要求的输出格式和详略。这些图片属于同一道题，按顺序阅读，重叠内容不要重复计算。'},
            ]},
            *self._text_history(history),
        ]
        yield from self._stream(messages, proxies)

    def _stream(self, messages, proxies):
        yield {'status': 'started', 'content': ''}
        started = time.monotonic()
        try:
            with self._create_client(proxies) as client:
                print(f"百炼请求开始: model={self.get_model_identifier()}, tier={self.reasoning_tier}", flush=True)
                answer = ''
                reasoning = ''
                answering = False
                first_content = False
                with client.chat.completions.create(
                    model=self.get_model_identifier(), messages=messages,
                    temperature=self.temperature, stream=True,
                    max_tokens=self._get_max_tokens(), **self._reasoning_kwargs()
                ) as response:
                    print(f"百炼收到响应头: {time.monotonic() - started:.2f}s", flush=True)
                    for chunk in response:
                        if not chunk.choices:
                            continue
                        delta = chunk.choices[0].delta
                        thought = getattr(delta, 'reasoning_content', None)
                        if not first_content and (delta.content or thought):
                            first_content = True
                            print(f"百炼首个内容: {time.monotonic() - started:.2f}s", flush=True)
                        if self._is_thinking_model() and thought:
                            reasoning += thought
                            yield {'status': 'reasoning', 'content': reasoning, 'is_reasoning': True}
                        if delta.content:
                            if reasoning and not answering:
                                yield {'status': 'reasoning_complete', 'content': reasoning, 'is_reasoning': True}
                            answering = True
                            answer += delta.content
                            yield {'status': 'streaming', 'content': answer}
                if answer.strip():
                    yield {'status': 'completed', 'content': answer}
                else:
                    yield {'status': 'error', 'error': '百炼返回空答案（可能只有思考内容）。请重试或切换模型。'}
                print(f"百炼请求结束: {time.monotonic() - started:.2f}s", flush=True)
        except Exception as exc:
            error = ('百炼请求超时：连接超过10秒或等待响应超过90秒。请重试或切换模型。'
                     if isinstance(exc, APITimeoutError) else str(exc))
            yield {'status': 'error', 'error': error}

    def _get_max_tokens(self) -> int:
        return getattr(self, 'max_tokens', None) or 4000

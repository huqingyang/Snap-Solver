import os
from typing import Generator, Dict, Optional, Any
from openai import OpenAI
from .base import BaseModel

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
        effort_map = {'fast': 'none', 'deep': 'medium', 'max': 'xhigh'}
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
        """Stream the model's response for text analysis"""
        try:
            # Initial status
            yield {"status": "started", "content": ""}

            # Save original environment state
            original_env = {
                'http_proxy': os.environ.get('http_proxy'),
                'https_proxy': os.environ.get('https_proxy')
            }

            try:
                # Set proxy environment variables if provided
                if proxies:
                    if 'http' in proxies:
                        os.environ['http_proxy'] = proxies['http']
                    if 'https' in proxies:
                        os.environ['https_proxy'] = proxies['https']

                # Initialize OpenAI compatible client for DashScope
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                )

                # Prepare messages
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": text}]
                    }
                ]

                # 创建聊天完成请求
                response = client.chat.completions.create(
                    model=self.get_model_identifier(),
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                    max_tokens=self._get_max_tokens(),
                    **self._reasoning_kwargs()
                )

                # 记录思考过程和回答
                reasoning_content = ""
                answer_content = ""
                is_answering = False

                # 当前档位/模型是否会产生思考过程
                has_thinking = self._is_thinking_model()
                print(f"分析文本使用模型标识符: {self.get_model_identifier()}, 是否含思考过程: {has_thinking}")

                for chunk in response:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # 处理思考过程
                    if has_thinking and hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        reasoning_content += delta.reasoning_content
                        # 思考过程作为一个独立的内容发送
                        yield {
                            "status": "reasoning",
                            "content": reasoning_content,
                            "is_reasoning": True
                        }
                    elif delta.content:
                        # 判断是否开始回答（从思考过程切换到回答）
                        if not is_answering and has_thinking:
                            is_answering = True
                            # 发送完整的思考过程
                            if reasoning_content:
                                yield {
                                    "status": "reasoning_complete",
                                    "content": reasoning_content,
                                    "is_reasoning": True
                                }
                        
                        # 累积回答内容
                        answer_content += delta.content
                        
                        # 发送回答内容
                        yield {
                            "status": "streaming",
                            "content": answer_content
                        }

                # 确保发送最终完整内容
                if answer_content:
                    yield {
                        "status": "completed",
                        "content": answer_content
                    }

            finally:
                # Restore original environment state
                for key, value in original_env.items():
                    if value is None:
                        if key in os.environ:
                            del os.environ[key]
                    else:
                        os.environ[key] = value

        except Exception as e:
            yield {
                "status": "error",
                "error": str(e)
            }

    def analyze_image(self, image_data: str, proxies: dict = None, history: list = None) -> Generator[dict, None, None]:
        """Stream model's response for image analysis"""
        try:
            # Initial status
            yield {"status": "started", "content": ""}

            # Save original environment state
            original_env = {
                'http_proxy': os.environ.get('http_proxy'),
                'https_proxy': os.environ.get('https_proxy')
            }

            try:
                # Set proxy environment variables if provided
                if proxies:
                    if 'http' in proxies:
                        os.environ['http_proxy'] = proxies['http']
                    if 'https' in proxies:
                        os.environ['https_proxy'] = proxies['https']

                # Initialize OpenAI compatible client for DashScope
                client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
                )

                # 使用系统提供的系统提示词，不再自动添加语言指令
                system_prompt = self.system_prompt

                # Prepare messages with image
                messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}]
                    },
                    {
                        "role": "user",
                        "content": [
                            *self._image_content(image_data),
                            {
                                "type": "text",
                                "text": "请结合全部图片作答，遵循系统提示词要求的输出格式和详略。这些图片属于同一道题，按顺序阅读，重叠内容不要重复计算。"
                            }
                        ]
                    }
                ]

                # 同题追问：既往问答与新追问追加在带图首轮之后
                # （DashScope 兼容模式与 OpenAI 一致，纯文本轮次用字符串 content）
                messages.extend(self._text_history(history))

                # 创建聊天完成请求
                response = client.chat.completions.create(
                    model=self.get_model_identifier(),
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                    max_tokens=self._get_max_tokens(),
                    **self._reasoning_kwargs()
                )

                # 记录思考过程和回答
                reasoning_content = ""
                answer_content = ""
                is_answering = False

                # 当前档位/模型是否会产生思考过程
                has_thinking = self._is_thinking_model()
                print(f"分析图像使用模型标识符: {self.get_model_identifier()}, 是否含思考过程: {has_thinking}")

                for chunk in response:
                    if not chunk.choices:
                        continue

                    delta = chunk.choices[0].delta

                    # 处理思考过程
                    if has_thinking and hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                        reasoning_content += delta.reasoning_content
                        # 思考过程作为一个独立的内容发送
                        yield {
                            "status": "reasoning",
                            "content": reasoning_content,
                            "is_reasoning": True
                        }
                    elif delta.content:
                        # 判断是否开始回答（从思考过程切换到回答）
                        if not is_answering and has_thinking:
                            is_answering = True
                            # 发送完整的思考过程
                            if reasoning_content:
                                yield {
                                    "status": "reasoning_complete",
                                    "content": reasoning_content,
                                    "is_reasoning": True
                                }
                        
                        # 累积回答内容
                        answer_content += delta.content
                        
                        # 发送回答内容
                        yield {
                            "status": "streaming",
                            "content": answer_content
                        }

                # 确保发送最终完整内容
                if answer_content:
                    yield {
                        "status": "completed",
                        "content": answer_content
                    }

            finally:
                # Restore original environment state
                for key, value in original_env.items():
                    if value is None:
                        if key in os.environ:
                            del os.environ[key]
                    else:
                        os.environ[key] = value

        except Exception as e:
            yield {
                "status": "error",
                "error": str(e)
            } 

    def _get_max_tokens(self) -> int:
        """返回合适的 max_tokens 值"""
        return self.max_tokens if hasattr(self, 'max_tokens') and self.max_tokens else 4000

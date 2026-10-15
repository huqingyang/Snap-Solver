from abc import ABC, abstractmethod
from typing import Generator, Any

# 统一推理档位：所有模型对外只暴露这三档，各子类内部映射到自家原生参数
REASONING_TIERS = ("fast", "deep", "max")


class BaseModel(ABC):
    def __init__(self, api_key: str, temperature: float = 0.7, system_prompt: str = None, language: str = None, api_base_url: str = None, reasoning_tier: str = "deep"):
        self.api_key = api_key
        self.temperature = temperature
        self.language = language
        self.system_prompt = system_prompt or self.get_default_system_prompt()
        self.api_base_url = api_base_url
        # 统一推理档位，非法值回退到 deep
        self.reasoning_tier = reasoning_tier if reasoning_tier in REASONING_TIERS else "deep"

    @abstractmethod
    def analyze_image(self, image_data: str, proxies: dict = None, history: list = None) -> Generator[dict, None, None]:
        """
        Analyze the given image and yield response chunks.

        Args:
            image_data: Base64 encoded image data
            proxies: Optional proxy configuration
            history: Optional follow-up turns appended after the image message,
                each {'role': 'user'|'assistant', 'content': str}; the last
                entry is the new user question (同题追问)

        Yields:
            dict: Response chunks with status and content
        """
        pass

    @staticmethod
    def _image_list(image_data) -> list[str]:
        """统一单图/多图输入，保留图片顺序。"""
        images = image_data if isinstance(image_data, list) else [image_data]
        if not images or any(not isinstance(item, str) or not item for item in images):
            raise ValueError("图像数据不能为空")
        return [item.split(',', 1)[1] if item.startswith('data:image') else item for item in images]

    @staticmethod
    def _image_mime(image: str) -> str:
        return 'image/png' if image.startswith('iVBOR') else 'image/jpeg'

    @classmethod
    def _image_content(cls, image_data) -> list[dict]:
        """OpenAI 兼容接口的图片内容块，单图与多图使用相同格式。"""
        return [
            {'type': 'image_url', 'image_url': {
                'url': f'data:{cls._image_mime(image)};base64,{image}'
            }}
            for image in cls._image_list(image_data)
        ]

    @staticmethod
    def _text_history(history) -> list:
        """清洗追问历史：只保留 user/assistant 的非空纯文本轮次，其余丢弃。
        各子类在拼完首条带图消息后，把返回值按自家消息格式追加进 messages。"""
        turns = []
        for turn in history or []:
            if not isinstance(turn, dict):
                continue
            role = turn.get('role')
            content = turn.get('content')
            if role in ('user', 'assistant') and isinstance(content, str) and content.strip():
                turns.append({'role': role, 'content': content})
        return turns

    @abstractmethod
    def analyze_text(self, text: str, proxies: dict = None) -> Generator[dict, None, None]:
        """
        Analyze the given text and yield response chunks.

        Args:
            text: Text to analyze
            proxies: Optional proxy configuration

        Yields:
            dict: Response chunks with status and content
        """
        pass

    def _apply_reasoning_tier(self, payload: dict) -> None:
        """将统一推理档位（self.reasoning_tier）映射为该家 API 的原生推理参数，
        原地修改 payload。基类默认为空实现——非推理模型无需任何推理参数，
        各支持推理的子类按自家 API 形状覆盖此方法。"""
        return None

    def get_default_system_prompt(self) -> str:
        """返回默认的系统提示词，子类可覆盖但不再是必须实现的方法"""
        return "您是一位专业的问题解决专家。请逐步分析问题，找出问题所在，并提供详细的解决方案。始终使用用户偏好的语言回答。"

    @abstractmethod
    def get_model_identifier(self) -> str:
        """Return the model identifier used in API calls"""
        pass

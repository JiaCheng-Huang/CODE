import abc
import os
from typing import Optional

from openai import OpenAI, APIError


class LLMClient(abc.ABC):

    @abc.abstractmethod
    def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            **kwargs
    ) -> str:
        pass


class OpenAICompatibleClient(LLMClient):
    base_url: Optional[str] = None
    default_model: str = "gpt-3.5-turbo"
    env_key: str = "OPENAI_API_KEY"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get(self.env_key)
        if not self.api_key:
            raise ValueError(f"{self.env_key} 未设置")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        temperature = kwargs.get("temperature", 0)
        seed = kwargs.get("seed", 42)
        model = kwargs.get("model", self.default_model)
        top_p = kwargs.get("top_p", 1)
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                top_p=top_p,
                n=1,
                stream=False,
            )
            return response.choices[0].message.content
        except APIError as e:
            raise RuntimeError(f"API 调用失败: {e}") from e


# ---------- DeepSeek ----------
class DeepSeekClient(OpenAICompatibleClient):
    base_url = "https://api.deepseek.com/v1"
    default_model = "deepseek-flash"  # DeepSeek-V4.1-Flash
    env_key = "DEEPSEEK_API_KEY"


# ---------- OpenAI GPT ----------
class OpenAIClient(OpenAICompatibleClient):
    base_url = None
    default_model = "gpt-3.5-turbo"
    env_key = "OPENAI_API_KEY"


# ---------- Google Gemini ----------
class GeminiClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None):
        import google.generativeai as genai
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 未设置")
        genai.configure(api_key=self.api_key)
        self.default_model = "gemini-1.5-flash"
        self.model_cache = {}

    def _get_model(self, model_name: str):
        import google.generativeai as genai
        if model_name not in self.model_cache:
            self.model_cache[model_name] = genai.GenerativeModel(model_name)
        return self.model_cache[model_name]

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        model_name = kwargs.get("model", self.default_model)
        model = self._get_model(model_name)
        generation_config = {
            "temperature": kwargs.get("temperature", 0),
            "max_output_tokens": kwargs.get("max_tokens"),
        }
        generation_config = {k: v for k, v in generation_config.items() if v is not None}
        try:
            if system_prompt:
                response = model.generate_content(
                    contents=prompt,
                    generation_config=generation_config,
                    system_instruction=system_prompt
                )
            else:
                response = model.generate_content(
                    contents=prompt,
                    generation_config=generation_config
                )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API 调用失败: {e}") from e


_PROVIDER_MAP = {
    "deepseek": DeepSeekClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def create_client(provider: str, api_key: Optional[str] = None) -> LLMClient:
    provider = provider.lower()
    client_class = _PROVIDER_MAP.get(provider)
    if client_class is None:
        raise ValueError(f"不支持的 provider: {provider}，可选: {list(_PROVIDER_MAP.keys())}")
    return client_class(api_key)


if __name__ == "__main__":
    client = create_client("deepseek")
    print(client.generate("什么是深度学习？", system_prompt="用一句话回答"))


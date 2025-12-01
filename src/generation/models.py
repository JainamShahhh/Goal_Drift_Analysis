import os
from abc import ABC, abstractmethod
from typing import List, Optional
import openai
import anthropic
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class ModelGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> List[str]:
        """
        Generates n completions for the given prompt.
        """
        pass

import time
import random

def retry_with_backoff(func, retries=3, initial_delay=1):
    """
    Simple retry helper with exponential backoff.
    """
    def wrapper(*args, **kwargs):
        delay = initial_delay
        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i == retries - 1:
                    print(f"Failed after {retries} retries: {e}")
                    raise e
                print(f"Error: {e}. Retrying in {delay}s...")
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
        return []
    return wrapper

class OpenAIGenerator(ModelGenerator):
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> List[str]:
        @retry_with_backoff
        def _call_api():
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                n=n,
                temperature=temperature,
            )
            return [choice.message.content for choice in response.choices]
        
        try:
            return _call_api()
        except:
            return []

class AnthropicGenerator(ModelGenerator):
    def __init__(self, model_name: str = "claude-3-haiku-20240307"):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> List[str]:
        results = []
        for _ in range(n):
            @retry_with_backoff
            def _call_api():
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text

            try:
                res = _call_api()
                results.append(res)
                # Small delay to be nice to the API
                time.sleep(0.5)
            except:
                pass
        return results

class GeminiGenerator(ModelGenerator):
    def __init__(self, model_name: str = "gemini-pro"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate(self, prompt: str, n: int = 1, temperature: float = 0.7) -> List[str]:
        results = []
        for _ in range(n):
            @retry_with_backoff
            def _call_api():
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=temperature
                    )
                )
                return response.text

            try:
                res = _call_api()
                results.append(res)
                time.sleep(0.5)
            except:
                pass
        return results

def get_model(provider: str, model_name: str) -> ModelGenerator:
    if provider == "openai":
        return OpenAIGenerator(model_name)
    elif provider == "anthropic":
        return AnthropicGenerator(model_name)
    elif provider == "google":
        return GeminiGenerator(model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")

"""
LLM 서비스 - 다양한 LLM 제공자 지원
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import json
import requests
import time

from utils.logger import get_logger
from utils.error_handler import retry_on_error
from core.exceptions import LLMAPIError

logger = get_logger(__name__)


class LLMProvider(ABC):
    """LLM 제공자 인터페이스"""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        텍스트 생성

        Args:
            prompt: 입력 프롬프트
            **kwargs: 추가 파라미터

        Returns:
            생성된 텍스트
        """
        pass

    def validate_response(self, response: str) -> bool:
        """응답 검증 (선택적)"""
        return bool(response and len(response) > 0)


class OllamaLLMProvider(LLMProvider):
    """
    Ollama 로컬 LLM 구현

    지원 모델:
    - mistral (추천)
    - llama3.1
    - qwen2.5:7b
    - phi3
    - gemma2:9b
    """

    def __init__(
        self,
        model: str = "mistral",
        base_url: str = "http://localhost:11434",
        timeout: int = 3600,  # 1시간 (대량 리뷰 배치 처리 대비)
        cache_service: Optional['CacheService'] = None
    ):
        """
        Args:
            model: 사용할 모델 이름
            base_url: Ollama 서버 URL
            timeout: 타임아웃 (초, 기본 3600초 = 1시간)
            cache_service: 캐시 서비스 인스턴스 (선택)
        """
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.timeout = timeout
        self.cache_service = cache_service

        logger.info(
            f"Ollama LLM Provider initialized",
            model=model,
            base_url=base_url,
            cache_enabled=cache_service is not None
        )

    @retry_on_error(
        max_retries=3,
        delay=1.0,
        backoff=2.0,
        exceptions=(requests.exceptions.RequestException, LLMAPIError)
    )
    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Ollama API로 텍스트 생성 (캐싱 + Retry 지원, Phase 3)

        Args:
            prompt: 입력 프롬프트
            max_tokens: 최대 생성 토큰 수
            temperature: 생성 다양성 (0.0-1.0)

        Returns:
            생성된 텍스트

        Raises:
            LLMAPIError: API 호출 실패 시 (3회 재시도 후)
        """
        # 캐시 체크 (Phase 3)
        if self.cache_service:
            cached_response = self.cache_service.get(prompt, model=self.model)
            if cached_response:
                return cached_response

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 40)
            }
        }

        try:
            logger.debug(f"Calling Ollama API", model=self.model)

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            generated_text = result.get("response", "")

            if not self.validate_response(generated_text):
                raise LLMAPIError("Empty response from Ollama")

            logger.info(
                f"Ollama API call successful",
                tokens_generated=result.get("eval_count", 0),
                duration_ms=result.get("total_duration", 0) / 1_000_000
            )

            # 캐시에 저장 (Phase 3)
            if self.cache_service:
                self.cache_service.set(prompt, generated_text, model=self.model)

            return generated_text

        except requests.exceptions.Timeout:
            raise LLMAPIError(
                f"Ollama API timeout after {self.timeout}s. "
                "Try a smaller model or increase timeout."
            )
        except requests.exceptions.ConnectionError:
            raise LLMAPIError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: 'ollama serve'"
            )
        except requests.exceptions.RequestException as e:
            raise LLMAPIError(f"Ollama API error: {str(e)}")
        except Exception as e:
            raise LLMAPIError(f"Unexpected error: {str(e)}")

    def check_health(self) -> bool:
        """
        Ollama 서버 상태 확인

        Returns:
            서버가 정상이면 True
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False


class ClaudeLLMProvider(LLMProvider):
    """Anthropic Claude API 구현"""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        """
        Args:
            api_key: Anthropic API 키
            model: Claude 모델 이름
        """
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
            self.model = model
            logger.info(f"Claude LLM Provider initialized", model=model)
        except ImportError:
            raise LLMAPIError(
                "anthropic package not installed. "
                "Install: pip install anthropic"
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Claude API로 텍스트 생성"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            generated_text = response.content[0].text

            logger.info(
                "Claude API call successful",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens
            )

            return generated_text

        except Exception as e:
            raise LLMAPIError(f"Claude API error: {str(e)}")


class OpenAILLMProvider(LLMProvider):
    """OpenAI GPT API 구현"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        """
        Args:
            api_key: OpenAI API 키
            model: GPT 모델 이름
        """
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
            self.model = model
            logger.info(f"OpenAI LLM Provider initialized", model=model)
        except ImportError:
            raise LLMAPIError(
                "openai package not installed. "
                "Install: pip install openai"
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """OpenAI API로 텍스트 생성"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )

            generated_text = response.choices[0].message.content

            logger.info(
                "OpenAI API call successful",
                tokens_used=response.usage.total_tokens
            )

            return generated_text

        except Exception as e:
            raise LLMAPIError(f"OpenAI API error: {str(e)}")


class LLMService:
    """
    LLM 서비스 - Provider 패턴으로 다양한 LLM 지원

    사용 예:
        # Ollama (로컬)
        provider = OllamaLLMProvider("mistral")
        service = LLMService(provider)

        # Claude (API)
        provider = ClaudeLLMProvider(api_key)
        service = LLMService(provider)
    """

    def __init__(self, provider: LLMProvider):
        """
        Args:
            provider: LLM 제공자 인스턴스
        """
        self.provider = provider
        self.logger = logger

    def generate(self, prompt: str, **kwargs) -> str:
        """
        텍스트 생성

        Args:
            prompt: 입력 프롬프트
            **kwargs: 추가 파라미터

        Returns:
            생성된 텍스트
        """
        return self.provider.generate(prompt, **kwargs)

    def generate_json(
        self,
        prompt: str,
        retries: int = 3,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        JSON 형식 응답 생성 (재시도 로직 포함, 개선된 디버깅)

        Args:
            prompt: 입력 프롬프트 (JSON 요청)
            retries: 재시도 횟수

        Returns:
            파싱된 JSON 딕셔너리 또는 None
        """
        for attempt in range(retries):
            try:
                self.logger.info(f"LLM API call attempt {attempt + 1}/{retries}")
                response = self.provider.generate(prompt, **kwargs)

                if not response:
                    self.logger.error(f"LLM returned empty response (attempt {attempt + 1}/{retries})")
                    continue

                # 응답 길이 로깅
                self.logger.info(f"LLM response received: {len(response)} characters")

                # JSON 추출 시도
                json_obj = self._extract_json(response)

                if json_obj:
                    self.logger.info(f"JSON parsing successful (attempt {attempt + 1}/{retries})")
                    return json_obj
                else:
                    # 파싱 실패 시 응답 일부 로깅
                    response_preview = response[:500] if len(response) > 500 else response
                    self.logger.warning(
                        f"JSON parsing failed (attempt {attempt + 1}/{retries}). "
                        f"Response preview: {response_preview}..."
                    )

            except Exception as e:
                self.logger.error(
                    f"Generate JSON error (attempt {attempt + 1}/{retries}): {str(e)}",
                    exc_info=True
                )

        self.logger.error("All JSON parsing attempts failed. Returning None.")
        return None

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        텍스트에서 JSON 추출 (개선된 nested JSON 지원)

        Args:
            text: LLM 응답 텍스트

        Returns:
            파싱된 JSON 또는 None
        """
        import re

        # 1. 직접 파싱 시도
        try:
            return json.loads(text)
        except:
            pass

        # 2. Markdown 코드 블록에서 추출 (```json ... ```)
        json_code_block = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if json_code_block:
            try:
                return json.loads(json_code_block.group(1))
            except:
                pass

        # 3. 일반 코드 블록에서 추출 (``` ... ```)
        code_block = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except:
                pass

        # 4. 중괄호 찾기 - 밸런스 체크하여 전체 JSON 추출
        brace_count = 0
        start_idx = -1

        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # 완전한 JSON 블록 발견
                    json_str = text[start_idx:i+1]
                    try:
                        return json.loads(json_str)
                    except:
                        # 첫 번째 JSON 블록 실패 시 다음 블록 찾기
                        start_idx = -1
                        continue

        return None


def create_llm_service(config: Dict[str, Any]) -> LLMService:
    """
    설정에서 LLM 서비스 생성 (팩토리 함수)

    Args:
        config: 설정 딕셔너리

        예시 (중첩 구조):
        {
            'provider': 'ollama',
            'ollama': {
                'model': 'gpt-oss:20b',
                'base_url': 'http://localhost:11434'
            },
            'cache_enabled': true
        }

        또는 (플랫 구조):
        {
            'provider': 'ollama',
            'model': 'mistral',
            'base_url': 'http://localhost:11434'
        }

    Returns:
        LLMService 인스턴스
    """
    # 캐시 서비스 생성 (Phase 3)
    cache_service = None
    if config.get('cache_enabled', True):  # 기본적으로 캐시 활성화
        from services.cache_service import CacheService
        cache_service = CacheService()

    provider_type = config.get('provider', 'ollama').lower()

    if provider_type == 'ollama':
        # 중첩 구조 지원 (config/llm_config.yaml)
        ollama_config = config.get('ollama', {})
        model = ollama_config.get('model') or config.get('model', 'mistral')
        base_url = ollama_config.get('base_url') or config.get('base_url', 'http://localhost:11434')
        timeout = ollama_config.get('timeout') or config.get('timeout', 3600)  # 1시간 기본값

        provider = OllamaLLMProvider(
            model=model,
            base_url=base_url,
            timeout=timeout,
            cache_service=cache_service
        )
    elif provider_type == 'claude':
        # 중첩 구조 지원
        claude_config = config.get('claude', {})
        api_key = claude_config.get('api_key') or config.get('api_key')

        if not api_key:
            raise ValueError("Claude API key required")

        model = claude_config.get('model') or config.get('model', 'claude-sonnet-4-20250514')

        provider = ClaudeLLMProvider(
            api_key=api_key,
            model=model
        )
    elif provider_type == 'openai':
        # 중첩 구조 지원
        openai_config = config.get('openai', {})
        api_key = openai_config.get('api_key') or config.get('api_key')

        if not api_key:
            raise ValueError("OpenAI API key required")

        model = openai_config.get('model') or config.get('model', 'gpt-4')

        provider = OpenAILLMProvider(
            api_key=api_key,
            model=model
        )
    else:
        raise ValueError(f"Unknown provider: {provider_type}")

    return LLMService(provider)

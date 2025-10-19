"""
LLM 서비스 단위 테스트 (Phase 4)
"""
import pytest
from unittest.mock import Mock, patch
from services.llm_service import (
    LLMService,
    OllamaLLMProvider,
    create_llm_service
)
from core.exceptions import LLMAPIError


@pytest.mark.unit
class TestLLMService:
    """LLMService 단위 테스트"""

    def test_service_initialization(self, mock_llm_service):
        """LLM Service 초기화 테스트"""
        assert mock_llm_service is not None

    @pytest.mark.fast
    def test_generate_text(self, mock_llm_service):
        """텍스트 생성 테스트 (모킹)"""
        prompt = "Analyze this review"

        response = mock_llm_service.generate(prompt)

        assert response == "Mocked LLM response"
        mock_llm_service.generate.assert_called_once_with(prompt)

    @pytest.mark.fast
    def test_generate_json(self, mock_llm_service):
        """JSON 생성 테스트 (모킹)"""
        prompt = "Extract pain points as JSON"

        result = mock_llm_service.generate_json(prompt)

        assert result is not None
        assert 'pain_points' in result
        assert isinstance(result['pain_points'], list)


@pytest.mark.unit
class TestOllamaLLMProvider:
    """OllamaLLMProvider 단위 테스트"""

    @pytest.fixture
    def provider(self):
        """Ollama Provider 인스턴스 (캐시 비활성화)"""
        return OllamaLLMProvider(
            model="gpt-oss:20b",
            base_url="http://localhost:11434",
            cache_service=None
        )

    def test_provider_initialization(self, provider):
        """Provider 초기화 테스트"""
        assert provider.model == "gpt-oss:20b"
        assert provider.base_url == "http://localhost:11434"
        assert provider.api_url == "http://localhost:11434/api/generate"

    @patch('services.llm_service.requests.post')
    def test_generate_success(self, mock_post, provider):
        """정상 API 호출 테스트 (모킹)"""
        # Mock 응답 설정
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'response': 'Generated text',
            'eval_count': 100,
            'total_duration': 5000000000
        }
        mock_post.return_value = mock_response

        result = provider.generate("Test prompt")

        assert result == "Generated text"
        mock_post.assert_called_once()

    @patch('services.llm_service.requests.post')
    def test_generate_timeout(self, mock_post, provider):
        """타임아웃 에러 처리 테스트"""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()

        with pytest.raises(LLMAPIError, match="timeout"):
            provider.generate("Test prompt")

    @patch('services.llm_service.requests.post')
    def test_generate_connection_error(self, mock_post, provider):
        """연결 에러 처리 테스트"""
        import requests
        mock_post.side_effect = requests.exceptions.ConnectionError()

        with pytest.raises(LLMAPIError, match="Cannot connect"):
            provider.generate("Test prompt")

    @patch('services.llm_service.requests.post')
    def test_retry_on_error(self, mock_post, provider):
        """재시도 로직 테스트"""
        import requests

        # 2번 실패 후 성공
        mock_post.side_effect = [
            requests.exceptions.RequestException("Temp error"),
            requests.exceptions.RequestException("Temp error"),
            Mock(status_code=200, json=lambda: {'response': 'Success'})
        ]

        result = provider.generate("Test prompt")

        assert result == "Success"
        assert mock_post.call_count == 3  # 3회 시도

    @patch('services.llm_service.requests.get')
    def test_health_check(self, mock_get, provider):
        """헬스 체크 테스트"""
        mock_get.return_value = Mock(status_code=200)

        is_healthy = provider.check_health()

        assert is_healthy is True

    @patch('services.llm_service.requests.get')
    def test_health_check_failure(self, mock_get, provider):
        """헬스 체크 실패 테스트"""
        mock_get.side_effect = Exception("Connection failed")

        is_healthy = provider.check_health()

        assert is_healthy is False


@pytest.mark.unit
class TestLLMServiceFactory:
    """LLM Service Factory 테스트"""

    def test_create_ollama_service(self):
        """Ollama Service 생성 테스트"""
        config = {
            'provider': 'ollama',
            'model': 'mistral',
            'base_url': 'http://localhost:11434',
            'cache_enabled': False
        }

        service = create_llm_service(config)

        assert service is not None
        assert isinstance(service, LLMService)

    def test_create_service_unknown_provider(self):
        """알 수 없는 Provider 에러 테스트"""
        config = {
            'provider': 'unknown_provider'
        }

        with pytest.raises(ValueError, match="Unknown provider"):
            create_llm_service(config)

    def test_json_extraction(self):
        """JSON 추출 테스트"""
        service = LLMService(Mock())

        # JSON 블록 추출
        text = '''
        Here is the result:
        ```json
        {"key": "value"}
        ```
        '''

        result = service._extract_json(text)

        assert result is not None
        assert result['key'] == 'value'

    def test_json_extraction_plain(self):
        """Plain JSON 추출 테스트"""
        service = LLMService(Mock())

        text = '{"name": "test", "count": 5}'

        result = service._extract_json(text)

        assert result is not None
        assert result['name'] == 'test'
        assert result['count'] == 5

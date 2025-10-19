"""
Pytest 설정 및 공통 Fixtures (Phase 4)

모든 테스트에서 사용할 공통 fixtures를 정의합니다.
"""
import pytest
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, MagicMock

# src 폴더를 Python path에 추가
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from services.llm_service import LLMService, OllamaLLMProvider
from core.orchestrator import ReviewAnalysisOrchestrator
from utils.logger import get_logger


@pytest.fixture
def sample_reviews_df():
    """샘플 리뷰 DataFrame (10개)"""
    data = {
        'review_id': [f'R{i}' for i in range(10)],
        'rating': [5, 4, 3, 2, 1, 5, 4, 3, 2, 1],
        'review_text': [
            'Great product! Battery lasts long.',
            'Good quality but price is high.',
            'Average battery life.',
            'Poor quality, battery dies fast.',
            'Terrible! Battery worst ever.',
            'Excellent build quality.',
            'Nice design but expensive.',
            'Okay for the price.',
            'Bad battery performance.',
            'Worst quality ever seen.'
        ],
        'verified_purchase': [True] * 10,
        'helpful_vote': [10, 8, 5, 3, 2, 12, 9, 4, 1, 0],
        'date': pd.date_range('2024-01-01', periods=10, freq='D')
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_reviews_json(tmp_path):
    """샘플 리뷰 JSON 파일"""
    reviews = [
        {
            'review_id': f'R{i}',
            'rating': [5, 4, 3, 2, 1][i % 5],
            'review_text': f'Sample review {i}',
            'verified_purchase': True,
            'helpful_vote': 10 - i,
            'date': f'2024-01-{i+1:02d}'
        }
        for i in range(20)
    ]

    filepath = tmp_path / "sample_reviews.json"
    with open(filepath, 'w') as f:
        json.dump(reviews, f)

    return str(filepath)


@pytest.fixture
def mock_llm_service():
    """Mock LLM Service (빠른 테스트용)"""
    mock_service = Mock(spec=LLMService)

    # generate() 메서드 모킹
    mock_service.generate.return_value = "Mocked LLM response"

    # generate_json() 메서드 모킹
    mock_service.generate_json.return_value = {
        "pain_points": [
            {
                "issue": "Battery drains quickly",
                "description": "Multiple users report short battery life",
                "frequency": "high",
                "severity": "high",
                "examples": ["Battery dies in 2 hours"]
            }
        ]
    }

    return mock_service


@pytest.fixture
def real_llm_service():
    """실제 LLM Service (통합 테스트용)"""
    config = {
        'provider': 'ollama',
        'model': 'gpt-oss:20b',
        'base_url': 'http://localhost:11434',
        'cache_enabled': False  # 테스트 시 캐시 비활성화
    }

    from services.llm_service import create_llm_service
    return create_llm_service(config)


@pytest.fixture
def test_config():
    """테스트용 설정"""
    return {
        'output_dir': 'output/test',
        'log_level': 'DEBUG',
        'cache_enabled': False,
        'llm': {
            'provider': 'ollama',
            'model': 'gpt-oss:20b',
            'base_url': 'http://localhost:11434'
        },
        'aspect_keywords_path': 'src/config/aspect_keywords/electronics.yaml'
    }


@pytest.fixture
def mock_orchestrator(test_config, mock_llm_service):
    """Mock Orchestrator (단위 테스트용)"""
    orchestrator = Mock(spec=ReviewAnalysisOrchestrator)
    orchestrator.config = test_config
    orchestrator.llm_service = mock_llm_service

    # run_analysis() 모킹
    orchestrator.run_analysis.return_value = {
        'basic_stats': {
            'total_reviews': 100,
            'avg_rating': 3.5
        },
        'sentiment_analysis': {
            'sentiment_distribution': {
                'positive': 40,
                'neutral': 30,
                'negative': 30
            }
        }
    }

    return orchestrator


@pytest.fixture
def aspect_keywords():
    """Aspect 키워드 딕셔너리"""
    return {
        'battery': ['battery', 'charge', 'charging', 'power', 'battery life'],
        'quality': ['quality', 'build', 'material', 'durable', 'durability'],
        'price': ['price', 'expensive', 'cheap', 'value', 'cost'],
        'design': ['design', 'look', 'style', 'appearance'],
        'performance': ['performance', 'speed', 'fast', 'slow']
    }


@pytest.fixture(autouse=True)
def cleanup_output(tmp_path, monkeypatch):
    """각 테스트 후 output 디렉토리 정리"""
    # 테스트용 임시 디렉토리 사용
    test_output = tmp_path / "test_output"
    test_output.mkdir(exist_ok=True)

    # 환경 변수로 output 디렉토리 변경
    monkeypatch.setenv('OUTPUT_DIR', str(test_output))

    yield test_output

    # 테스트 후 자동 정리 (필요 시)


@pytest.fixture
def logger():
    """테스트용 로거"""
    return get_logger('test')


# Pytest hooks
def pytest_configure(config):
    """Pytest 초기 설정"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for full pipeline"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 5 seconds"
    )
    config.addinivalue_line(
        "markers", "fast: Fast tests with mocked dependencies"
    )


def pytest_collection_modifyitems(config, items):
    """테스트 항목 수정"""
    for item in items:
        # integration 테스트에 자동으로 slow 마커 추가
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.slow)

        # unit 테스트에 fast 마커 추가 (LLM 모킹 시)
        if "unit" in item.keywords and "mock_llm" in item.fixturenames:
            item.add_marker(pytest.mark.fast)

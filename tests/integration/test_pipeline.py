"""
전체 파이프라인 통합 테스트 (Phase 4)

실제 Orchestrator를 사용하여 전체 분석 파이프라인을 테스트합니다.
"""
import pytest
import pandas as pd
from pathlib import Path
from core.orchestrator import ReviewAnalysisOrchestrator


@pytest.mark.integration
@pytest.mark.slow
class TestFullPipeline:
    """전체 파이프라인 통합 테스트"""

    @pytest.fixture
    def orchestrator(self, test_config):
        """실제 Orchestrator 인스턴스"""
        return ReviewAnalysisOrchestrator(
            config=test_config,
            llm_config_path='src/config/llm_config.yaml'
        )

    def test_full_pipeline_with_sample_data(self, orchestrator):
        """샘플 데이터로 전체 파이프라인 실행 테스트"""
        # fixtures/sample_reviews.json 사용
        data_path = 'tests/fixtures/sample_reviews.json'

        result = orchestrator.run_analysis(
            data_path=data_path,
            product_id='TEST_PRODUCT',
            limit=10,
            enable_llm=False  # 빠른 테스트를 위해 LLM 비활성화
        )

        # Stage 1-2: 기본 통계 검증
        assert 'basic_stats' in result
        assert result['basic_stats']['total_reviews'] > 0
        assert 'avg_rating' in result['basic_stats']

        # Stage 3: 감성 분석 검증
        assert 'sentiment_analysis' in result
        assert 'sentiment_distribution' in result['sentiment_analysis']
        assert 'absa' in result['sentiment_analysis']

        # 메타데이터 검증
        assert 'metadata' in result
        assert result['metadata']['product_id'] == 'TEST_PRODUCT'

    @pytest.mark.slow
    def test_pipeline_with_llm_enabled(self, orchestrator):
        """LLM 활성화한 전체 파이프라인 테스트"""
        data_path = 'tests/fixtures/sample_reviews.json'

        result = orchestrator.run_analysis(
            data_path=data_path,
            product_id='TEST_LLM',
            limit=10,
            enable_llm=True  # LLM 활성화
        )

        # LLM 기반 분석 검증
        assert 'insights' in result
        if result['insights']:
            assert 'pain_points' in result['insights']

        assert 'action_plan' in result
        if result['action_plan']:
            assert 'action_plan' in result['action_plan']

        assert 'final_report' in result

    def test_pipeline_stages_execution_order(self, orchestrator, monkeypatch):
        """파이프라인 Stage 실행 순서 테스트"""
        execution_log = []

        # 각 Stage 메서드를 모킹하여 실행 순서 추적
        original_load = orchestrator._load_and_preprocess_data

        def mock_load(*args, **kwargs):
            execution_log.append('stage_1_load')
            return original_load(*args, **kwargs)

        monkeypatch.setattr(orchestrator, '_load_and_preprocess_data', mock_load)

        # 간단한 실행
        orchestrator.run_analysis(
            data_path='tests/fixtures/sample_reviews.json',
            product_id='TEST_ORDER',
            limit=5,
            enable_llm=False
        )

        # 실행 순서 검증
        assert 'stage_1_load' in execution_log

    def test_pipeline_error_handling(self, orchestrator):
        """파이프라인 에러 처리 테스트"""
        # 존재하지 않는 파일
        with pytest.raises(Exception):
            orchestrator.run_analysis(
                data_path='nonexistent_file.json',
                product_id='ERROR_TEST',
                limit=10
            )

    def test_pipeline_output_structure(self, orchestrator):
        """파이프라인 출력 구조 검증"""
        result = orchestrator.run_analysis(
            data_path='tests/fixtures/sample_reviews.json',
            product_id='STRUCTURE_TEST',
            limit=10,
            enable_llm=False
        )

        # 필수 키 검증
        required_keys = [
            'basic_stats',
            'sentiment_analysis',
            'negative_reviews',
            'metadata'
        ]

        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # 메타데이터 구조 검증
        metadata = result['metadata']
        assert 'product_id' in metadata
        assert 'timestamp' in metadata
        assert 'version' in metadata

    @pytest.mark.parametrize("limit", [10, 50, 100])
    def test_pipeline_with_different_limits(self, orchestrator, limit):
        """다양한 리뷰 수 제한으로 테스트"""
        result = orchestrator.run_analysis(
            data_path='tests/fixtures/sample_reviews.json',
            product_id=f'LIMIT_{limit}',
            limit=limit,
            enable_llm=False
        )

        total_reviews = result['basic_stats']['total_reviews']
        assert total_reviews <= limit


@pytest.mark.integration
class TestVisualizationIntegration:
    """시각화 통합 테스트 (Phase 3)"""

    @pytest.fixture
    def orchestrator(self, test_config):
        """Orchestrator 인스턴스"""
        return ReviewAnalysisOrchestrator(
            config=test_config,
            llm_config_path='src/config/llm_config.yaml'
        )

    def test_visualization_generation(self, orchestrator, tmp_path):
        """시각화 차트 생성 테스트"""
        # 출력 디렉토리 설정
        orchestrator.config['output_dir'] = str(tmp_path)

        result = orchestrator.run_analysis(
            data_path='tests/fixtures/sample_reviews.json',
            product_id='VIZ_TEST',
            limit=20,
            enable_llm=False
        )

        # 시각화 결과 검증
        if 'visualizations' in result:
            viz = result['visualizations']
            assert 'charts' in viz
            assert viz['chart_count'] > 0

            # 생성된 파일 확인
            for chart_name, chart_path in viz['charts'].items():
                assert Path(chart_path).exists()


@pytest.mark.integration
class TestCacheIntegration:
    """캐시 통합 테스트 (Phase 3)"""

    @pytest.fixture
    def orchestrator_with_cache(self, test_config):
        """캐시 활성화한 Orchestrator"""
        config = test_config.copy()
        config['cache_enabled'] = True

        return ReviewAnalysisOrchestrator(
            config=config,
            llm_config_path='src/config/llm_config.yaml'
        )

    @pytest.mark.slow
    def test_cache_hit_on_repeat(self, orchestrator_with_cache):
        """반복 실행 시 캐시 히트 테스트"""
        prompt = "Test prompt for caching"

        # 1차 실행 (캐시 미스)
        result1 = orchestrator_with_cache.run_analysis(
            data_path='tests/fixtures/sample_reviews.json',
            product_id='CACHE_TEST',
            limit=10,
            enable_llm=True
        )

        # 2차 실행 (캐시 히트 예상)
        result2 = orchestrator_with_cache.run_analysis(
            data_path='tests/fixtures/sample_reviews.json',
            product_id='CACHE_TEST',
            limit=10,
            enable_llm=True
        )

        # 결과 일관성 검증 (캐시된 경우 동일 결과)
        # Note: 완전 동일하지 않을 수 있으므로 핵심 데이터만 비교
        assert result1['basic_stats'] == result2['basic_stats']

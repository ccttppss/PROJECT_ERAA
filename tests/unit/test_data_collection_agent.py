"""
데이터 수집 에이전트 단위 테스트 (Phase 4)
"""
import pytest
import pandas as pd
from agents.data_collection_agent import DataCollectionAgent


@pytest.mark.unit
class TestDataCollectionAgent:
    """DataCollectionAgent 단위 테스트"""

    @pytest.fixture
    def agent(self, test_config, logger):
        """에이전트 인스턴스"""
        return DataCollectionAgent(test_config, logger=logger)

    def test_agent_initialization(self, agent):
        """에이전트 초기화 테스트"""
        assert agent is not None
        assert agent.VERSION == "1.0.0"
        assert agent.config is not None

    def test_execute_with_valid_data(self, agent, sample_reviews_df):
        """정상 데이터로 실행 테스트"""
        input_data = {
            'reviews_df': sample_reviews_df,
            'product_id': 'TEST123'
        }

        result = agent.execute(input_data)

        # 기본 통계 검증
        assert 'basic_stats' in result
        assert result['basic_stats']['total_reviews'] == 10
        assert 'avg_rating' in result['basic_stats']
        assert 1 <= result['basic_stats']['avg_rating'] <= 5

        # DataFrame 검증
        assert 'processed_df' in result
        assert isinstance(result['processed_df'], pd.DataFrame)

    def test_execute_with_empty_data(self, agent):
        """빈 데이터로 실행 테스트"""
        input_data = {
            'reviews_df': pd.DataFrame(),
            'product_id': 'TEST123'
        }

        result = agent.execute(input_data)

        # 에러 처리 검증
        assert 'error' in result or result['basic_stats']['total_reviews'] == 0

    @pytest.mark.fast
    def test_calculate_basic_stats(self, agent, sample_reviews_df):
        """기본 통계 계산 테스트"""
        stats = agent._calculate_basic_stats(sample_reviews_df)

        assert stats['total_reviews'] == 10
        assert stats['avg_rating'] == pytest.approx(3.0, 0.1)
        assert 'rating_distribution' in stats
        assert stats['rating_distribution'][5] == 2
        assert stats['rating_distribution'][1] == 2

    def test_filter_negative_reviews(self, agent, sample_reviews_df):
        """부정 리뷰 필터링 테스트"""
        negative_df = agent._filter_negative_reviews(sample_reviews_df)

        # 별점 2 이하만 필터링
        assert len(negative_df) == 4  # ratings: 2, 1, 2, 1
        assert all(negative_df['rating'] <= 2)

    def test_calculate_review_length(self, agent, sample_reviews_df):
        """리뷰 길이 계산 테스트"""
        df_with_length = agent._add_review_features(sample_reviews_df)

        assert 'review_length' in df_with_length.columns
        assert all(df_with_length['review_length'] > 0)
        assert df_with_length.iloc[0]['review_length'] == len('Great product! Battery lasts long.')

    def test_helpful_ratio_calculation(self, agent, sample_reviews_df):
        """도움 비율 계산 테스트"""
        df_with_features = agent._add_review_features(sample_reviews_df)

        assert 'helpful_ratio' in df_with_features.columns
        # helpful_vote / (helpful_vote + 1) 계산 확인

    @pytest.mark.parametrize("rating,expected_sentiment", [
        (5, 'positive'),
        (4, 'positive'),
        (3, 'neutral'),
        (2, 'negative'),
        (1, 'negative')
    ])
    def test_rating_to_sentiment(self, agent, rating, expected_sentiment):
        """별점 → 감정 변환 테스트"""
        df = pd.DataFrame({'rating': [rating]})
        df_with_sentiment = agent._add_review_features(df)

        assert df_with_sentiment['sentiment'][0] == expected_sentiment

"""
감성 분석 에이전트 단위 테스트 (Phase 4)
"""
import pytest
import pandas as pd
from agents.sentiment_agent import SentimentAnalysisAgent


@pytest.mark.unit
class TestSentimentAnalysisAgent:
    """SentimentAnalysisAgent 단위 테스트"""

    @pytest.fixture
    def agent(self, test_config, logger):
        """에이전트 인스턴스"""
        return SentimentAnalysisAgent(test_config, logger=logger)

    def test_agent_initialization(self, agent):
        """에이전트 초기화 테스트"""
        assert agent is not None
        assert agent.VERSION == "1.0.0"
        assert hasattr(agent, 'aspect_keywords')

    def test_execute_with_valid_data(self, agent, sample_reviews_df):
        """정상 데이터로 실행 테스트"""
        input_data = {
            'reviews_df': sample_reviews_df
        }

        result = agent.execute(input_data)

        # 감성 분포 검증
        assert 'sentiment_distribution' in result
        sentiment_dist = result['sentiment_distribution']
        assert 'positive' in sentiment_dist
        assert 'neutral' in sentiment_dist
        assert 'negative' in sentiment_dist

        # ABSA 결과 검증
        assert 'absa' in result
        assert isinstance(result['absa'], dict)

    def test_sentiment_distribution(self, agent, sample_reviews_df):
        """감성 분포 계산 테스트"""
        distribution = agent._calculate_sentiment_distribution(sample_reviews_df)

        # 샘플 데이터: 5,4 (positive=4), 3 (neutral=2), 2,1 (negative=4)
        assert distribution['positive'] == 4
        assert distribution['neutral'] == 2
        assert distribution['negative'] == 4

    @pytest.mark.fast
    def test_aspect_extraction(self, agent):
        """Aspect 추출 테스트"""
        review_text = "Great battery life but expensive price"

        aspects = agent._extract_aspects(review_text)

        assert 'battery' in aspects
        assert 'price' in aspects
        assert 'quality' not in aspects

    def test_aspect_extraction_multiple_keywords(self, agent):
        """여러 키워드가 포함된 Aspect 추출"""
        review_text = "Battery and charging are both excellent"

        aspects = agent._extract_aspects(review_text)

        # 'battery'와 'charging' 모두 battery aspect의 키워드
        assert 'battery' in aspects

    def test_absa_analysis(self, agent, sample_reviews_df):
        """ABSA (Aspect-Based Sentiment Analysis) 테스트"""
        absa_result = agent._perform_absa(sample_reviews_df)

        assert isinstance(absa_result, dict)

        # battery aspect 확인
        if 'battery' in absa_result:
            battery_data = absa_result['battery']
            assert 'mention_count' in battery_data
            assert 'sentiment_ratio' in battery_data
            assert 'positive' in battery_data['sentiment_ratio']
            assert 'negative' in battery_data['sentiment_ratio']

    def test_aspect_summary_generation(self, agent):
        """Aspect 요약 생성 테스트"""
        absa_result = {
            'battery': {
                'mention_count': 5,
                'sentiment_ratio': {'positive': 0.4, 'neutral': 0.2, 'negative': 0.4},
                'avg_rating': 2.8
            },
            'quality': {
                'mention_count': 3,
                'sentiment_ratio': {'positive': 0.67, 'neutral': 0.33, 'negative': 0.0},
                'avg_rating': 4.3
            }
        }

        summary = agent._generate_aspect_summary(absa_result)

        assert len(summary) == 2
        assert summary[0]['aspect'] in ['battery', 'quality']

        # 언급 횟수 내림차순 정렬 확인
        assert summary[0]['mention_count'] >= summary[1]['mention_count']

    def test_empty_reviews(self, agent):
        """빈 리뷰 처리 테스트"""
        empty_df = pd.DataFrame(columns=['rating', 'review_text', 'sentiment'])

        result = agent.execute({'reviews_df': empty_df})

        assert result['sentiment_distribution']['positive'] == 0
        assert result['sentiment_distribution']['neutral'] == 0
        assert result['sentiment_distribution']['negative'] == 0

    @pytest.mark.parametrize("text,expected_aspects", [
        ("Battery is good", ['battery']),
        ("Quality and design are great", ['quality', 'design']),
        ("Price is too high", ['price']),
        ("No aspects mentioned here", []),
    ])
    def test_aspect_extraction_cases(self, agent, text, expected_aspects):
        """다양한 케이스 Aspect 추출 테스트"""
        extracted = agent._extract_aspects(text.lower())

        for aspect in expected_aspects:
            assert aspect in extracted

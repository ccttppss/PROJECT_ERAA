"""
감성 분석 에이전트 (Phase 2: ABSA 포함)

Phase 2 업그레이드:
- ABSA (Aspect-Based Sentiment Analysis) 구현
- 키워드 기반 aspect 추출
- aspect별 감정 분류
"""
import pandas as pd
import yaml
from typing import Dict, Any, List, Set
from pathlib import Path
from core.base_agent import BaseAgent


class SentimentAnalysisAgent(BaseAgent):
    """
    감성 분석 에이전트 (Phase 2)

    기능:
    1. 별점 기반 전체 감정 분류 (Phase 1)
    2. ABSA: aspect 추출 및 aspect별 감정 분류 (Phase 2)
    """

    VERSION = "2.0.0"  # Phase 2

    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)

        # Aspect keywords 로드
        self.aspect_keywords = self._load_aspect_keywords()
        self.logger.info(f"Loaded {len(self.aspect_keywords)} aspect categories")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        감성 분석 실행 (Phase 2)

        Args:
            input_data: {'dataframe': pd.DataFrame, ...}

        Returns:
            감성 분석 결과 + ABSA 결과
        """
        self._log_execution_start()

        df = input_data.get('dataframe')
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Invalid input: 'dataframe' required")

        # Phase 1: 별점 기반 감성 분류
        sentiment_counts = self._analyze_sentiment(df)

        # 감성별 평균 리뷰 길이
        sentiment_length = self._calculate_sentiment_metrics(df)

        # Phase 2: ABSA (Aspect-Based Sentiment Analysis)
        absa_results = self.aspect_based_sentiment(df)

        result = {
            "sentiment_distribution": sentiment_counts,
            "sentiment_metrics": sentiment_length,
            "total_analyzed": len(df),
            # Phase 2 추가
            "absa": absa_results,
            "aspect_summary": self._get_aspect_summary(absa_results)
        }

        self.log_metrics("sentiment_positive", sentiment_counts.get('positive', 0))
        self.log_metrics("sentiment_negative", sentiment_counts.get('negative', 0))
        self.log_metrics("aspects_detected", len(absa_results))

        self._log_execution_end()

        return result

    def _load_aspect_keywords(self) -> Dict[str, List[str]]:
        """
        aspect_keywords YAML 파일 로드

        Returns:
            {aspect_name: [keywords]}
        """
        # YAML 파일 경로
        yaml_path = Path(__file__).parent.parent / "config" / "aspect_keywords" / "electronics.yaml"

        if not yaml_path.exists():
            self.logger.warning(f"Aspect keywords file not found: {yaml_path}")
            return {}

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            aspects = data.get('aspects', {})

            # {aspect: [keywords]} 형태로 변환
            aspect_keywords = {}
            for aspect_name, aspect_data in aspects.items():
                keywords = aspect_data.get('keywords', [])
                aspect_keywords[aspect_name] = [kw.lower() for kw in keywords]

            return aspect_keywords

        except Exception as e:
            self.logger.error(f"Failed to load aspect keywords: {e}")
            return {}

    def extract_aspects(self, review_text: str) -> Set[str]:
        """
        리뷰 텍스트에서 언급된 aspect 추출 (키워드 매칭)

        Args:
            review_text: 리뷰 텍스트

        Returns:
            Set of aspect names mentioned in the review
        """
        if not review_text or not isinstance(review_text, str):
            return set()

        review_lower = review_text.lower()
        mentioned_aspects = set()

        for aspect_name, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                if keyword in review_lower:
                    mentioned_aspects.add(aspect_name)
                    break  # 이 aspect는 발견됨, 다음 aspect로

        return mentioned_aspects

    def aspect_based_sentiment(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Aspect-Based Sentiment Analysis (ABSA)

        각 aspect에 대해:
        - 언급 횟수
        - 긍정/부정/중립 분포
        - 평균 별점

        Args:
            df: 리뷰 DataFrame (reviewText, overall, sentiment_label 필요)

        Returns:
            {
                aspect_name: {
                    "mention_count": int,
                    "positive": int,
                    "negative": int,
                    "neutral": int,
                    "avg_rating": float,
                    "sentiment_ratio": {"positive": %, "negative": %, ...}
                }
            }
        """
        if self.aspect_keywords is None or len(self.aspect_keywords) == 0:
            self.logger.warning("No aspect keywords loaded, skipping ABSA")
            return {}

        # 각 리뷰에서 aspect 추출
        df['aspects'] = df['reviewText'].apply(self.extract_aspects)

        # 각 aspect별로 통계 수집
        absa_results = {}

        for aspect_name in self.aspect_keywords.keys():
            # 이 aspect를 언급한 리뷰 필터링
            aspect_reviews = df[df['aspects'].apply(lambda x: aspect_name in x)]

            if len(aspect_reviews) == 0:
                continue  # 언급 없으면 스킵

            # 감정 분포 계산
            sentiment_counts = aspect_reviews['sentiment_label'].value_counts().to_dict()

            total_mentions = len(aspect_reviews)

            absa_results[aspect_name] = {
                "mention_count": total_mentions,
                "positive": sentiment_counts.get('positive', 0),
                "negative": sentiment_counts.get('negative', 0),
                "neutral": sentiment_counts.get('neutral', 0),
                "avg_rating": float(aspect_reviews['overall'].mean()),
                "sentiment_ratio": {
                    "positive": round(sentiment_counts.get('positive', 0) / total_mentions * 100, 1),
                    "negative": round(sentiment_counts.get('negative', 0) / total_mentions * 100, 1),
                    "neutral": round(sentiment_counts.get('neutral', 0) / total_mentions * 100, 1)
                }
            }

        self.logger.info(f"ABSA completed: {len(absa_results)} aspects detected")

        return absa_results

    def _get_aspect_summary(self, absa_results: Dict) -> List[Dict]:
        """
        ABSA 결과를 요약 (상위 aspect만)

        Returns:
            상위 aspect 리스트 (언급 횟수 순)
        """
        if not absa_results:
            return []

        # 언급 횟수 순으로 정렬
        sorted_aspects = sorted(
            absa_results.items(),
            key=lambda x: x[1]['mention_count'],
            reverse=True
        )

        summary = []
        for aspect_name, data in sorted_aspects[:5]:  # 상위 5개만
            summary.append({
                "aspect": aspect_name,
                "mentions": data['mention_count'],
                "avg_rating": round(data['avg_rating'], 2),
                "dominant_sentiment": self._get_dominant_sentiment(data)
            })

        return summary

    def _get_dominant_sentiment(self, aspect_data: Dict) -> str:
        """aspect의 주요 감정 판단"""
        positive = aspect_data.get('positive', 0)
        negative = aspect_data.get('negative', 0)
        neutral = aspect_data.get('neutral', 0)

        max_count = max(positive, negative, neutral)

        if max_count == positive:
            return "positive"
        elif max_count == negative:
            return "negative"
        else:
            return "neutral"

    def _analyze_sentiment(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        별점 기반 감성 분류 (Phase 1 기능 유지)

        Args:
            df: 리뷰 데이터프레임

        Returns:
            감성별 카운트
        """
        if 'sentiment_label' in df.columns:
            counts = df['sentiment_label'].value_counts().to_dict()
        else:
            # sentiment_label이 없으면 overall에서 직접 계산
            counts = {
                'positive': int((df['overall'] >= 4).sum()),
                'negative': int((df['overall'] <= 2).sum()),
                'neutral': int(((df['overall'] > 2) & (df['overall'] < 4)).sum())
            }

        self.logger.info(
            "Sentiment analysis completed",
            positive=counts.get('positive', 0),
            negative=counts.get('negative', 0),
            neutral=counts.get('neutral', 0)
        )

        return counts

    def _calculate_sentiment_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        감성별 메트릭 계산 (Phase 1 기능 유지)

        Args:
            df: 리뷰 데이터프레임

        Returns:
            감성별 메트릭
        """
        metrics = {}

        if 'sentiment_label' not in df.columns:
            # sentiment_label 생성
            df['sentiment_label'] = df['overall'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
            )

        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_df = df[df['sentiment_label'] == sentiment]

            if len(sentiment_df) > 0:
                metrics[sentiment] = {
                    "count": len(sentiment_df),
                    "avg_length": int(sentiment_df['review_length'].mean()) if 'review_length' in sentiment_df.columns else 0,
                    "avg_rating": float(sentiment_df['overall'].mean())
                }
            else:
                metrics[sentiment] = {
                    "count": 0,
                    "avg_length": 0,
                    "avg_rating": 0.0
                }

        return metrics

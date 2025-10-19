"""
데이터 수집 및 기본 통계 분석 에이전트
"""
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from core.base_agent import BaseAgent
from core.exceptions import InsufficientDataError


class DataCollectionAgent(BaseAgent):
    """리뷰 수집 및 기본 분석 에이전트"""

    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.df: Optional[pd.DataFrame] = None
        self.stats: Dict[str, Any] = {}

    def execute(self, input_data: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터 수집 및 기본 통계 분석 실행

        Args:
            input_data: 전처리된 리뷰 데이터프레임

        Returns:
            기본 통계 및 필터링된 데이터
        """
        self._log_execution_start()

        if not self.validate_input(input_data):
            raise InsufficientDataError("Invalid or insufficient data")

        self.df = input_data.copy()

        # 기본 통계 수집
        self.stats = self.collect_basic_stats()

        # 메트릭 로깅
        self.log_metrics("total_reviews", self.stats['total_reviews'])
        self.log_metrics("avg_rating", self.stats['avg_rating'])

        self._log_execution_end()

        return {
            "stats": self.stats,
            "dataframe": self.df,
            "negative_reviews": self.get_negative_reviews(),
            "positive_reviews": self.get_positive_reviews(),
            "recent_reviews": self.get_recent_reviews()
        }

    def validate_input(self, input_data: Any) -> bool:
        """입력 데이터 검증"""
        if not isinstance(input_data, pd.DataFrame):
            self.logger.error("Input must be a pandas DataFrame")
            return False

        if len(input_data) == 0:
            self.logger.error("DataFrame is empty")
            return False

        required_columns = ['reviewText', 'overall']
        missing = [col for col in required_columns if col not in input_data.columns]
        if missing:
            self.logger.error(f"Missing required columns: {missing}")
            return False

        return True

    def collect_basic_stats(self) -> Dict[str, Any]:
        """
        기본 통계 수집

        Returns:
            통계 딕셔너리
        """
        if self.df is None:
            return {}

        stats = {
            'total_reviews': len(self.df),
            'avg_rating': float(self.df['overall'].mean()),
            'rating_distribution': self.df['overall'].value_counts().to_dict(),
            'avg_review_length': float(self.df['review_length'].mean()) if 'review_length' in self.df.columns else 0
        }

        # 날짜 범위 (date 컬럼이 있는 경우)
        if 'date' in self.df.columns:
            stats['date_range'] = {
                'start': str(self.df['date'].min()),
                'end': str(self.df['date'].max())
            }

        # 감성 분포
        if 'sentiment_label' in self.df.columns:
            stats['sentiment_distribution'] = self.df['sentiment_label'].value_counts().to_dict()

        self.logger.info(
            "Basic statistics collected",
            total=stats['total_reviews'],
            avg_rating=round(stats['avg_rating'], 2)
        )

        return stats

    def filter_reviews(
        self,
        min_rating: Optional[float] = None,
        max_rating: Optional[float] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> pd.DataFrame:
        """
        조건에 맞는 리뷰 필터링

        Args:
            min_rating: 최소 평점
            max_rating: 최대 평점
            date_from: 시작 날짜
            date_to: 종료 날짜

        Returns:
            필터링된 데이터프레임
        """
        if self.df is None:
            return pd.DataFrame()

        filtered = self.df.copy()

        if min_rating is not None:
            filtered = filtered[filtered['overall'] >= min_rating]

        if max_rating is not None:
            filtered = filtered[filtered['overall'] <= max_rating]

        if date_from and 'date' in filtered.columns:
            filtered = filtered[filtered['date'] >= pd.to_datetime(date_from)]

        if date_to and 'date' in filtered.columns:
            filtered = filtered[filtered['date'] <= pd.to_datetime(date_to)]

        self.logger.info(
            f"Filtered reviews",
            original_count=len(self.df),
            filtered_count=len(filtered)
        )

        return filtered

    def get_negative_reviews(self, threshold: float = 2.0) -> pd.DataFrame:
        """
        부정적 리뷰만 추출

        Args:
            threshold: 부정 리뷰 임계값 (이하)

        Returns:
            부정 리뷰 데이터프레임
        """
        if self.df is None:
            return pd.DataFrame()

        negative = self.df[self.df['overall'] <= threshold].copy()

        self.logger.info(
            f"Extracted negative reviews",
            count=len(negative),
            percentage=round(len(negative) / len(self.df) * 100, 2)
        )

        return negative

    def get_positive_reviews(self, threshold: float = 4.0) -> pd.DataFrame:
        """
        긍정적 리뷰만 추출

        Args:
            threshold: 긍정 리뷰 임계값 (이상)

        Returns:
            긍정 리뷰 데이터프레임
        """
        if self.df is None:
            return pd.DataFrame()

        positive = self.df[self.df['overall'] >= threshold].copy()

        self.logger.info(
            f"Extracted positive reviews",
            count=len(positive),
            percentage=round(len(positive) / len(self.df) * 100, 2)
        )

        return positive

    def get_recent_reviews(self, days: int = 90) -> pd.DataFrame:
        """
        최근 n일 리뷰 추출

        Args:
            days: 기간 (일)

        Returns:
            최근 리뷰 데이터프레임
        """
        if self.df is None or 'date' not in self.df.columns:
            return pd.DataFrame()

        cutoff_date = self.df['date'].max() - pd.Timedelta(days=days)
        recent = self.df[self.df['date'] >= cutoff_date].copy()

        self.logger.info(
            f"Extracted recent reviews ({days} days)",
            count=len(recent)
        )

        return recent

    def get_top_helpful_reviews(self, n: int = 10) -> pd.DataFrame:
        """
        가장 도움이 된 리뷰 추출

        Args:
            n: 추출할 리뷰 수

        Returns:
            도움됨 순으로 정렬된 데이터프레임
        """
        if self.df is None or 'helpful_ratio' not in self.df.columns:
            return pd.DataFrame()

        top = self.df.nlargest(n, 'helpful_ratio').copy()

        return top

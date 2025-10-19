"""
리뷰 데이터 전처리 모듈
"""
import re
import pandas as pd
from typing import Optional
from utils.logger import get_logger
from core.exceptions import DataValidationError

logger = get_logger(__name__)


class DataPreprocessor:
    """리뷰 데이터 전처리 클래스"""

    def __init__(self):
        self.logger = logger

    def clean_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        텍스트 정제

        Args:
            df: 원본 데이터프레임

        Returns:
            정제된 데이터프레임
        """
        df = df.copy()

        # reviewText가 없는 경우 처리
        if 'reviewText' not in df.columns:
            raise DataValidationError("'reviewText' column not found")

        # HTML 태그 제거
        df['reviewText'] = df['reviewText'].astype(str).str.replace(
            '<[^<]+?>',
            '',
            regex=True
        )

        # 특수문자 정규화 (구두점은 유지)
        df['reviewText'] = df['reviewText'].str.replace(
            '[^\w\s.,!?-]',
            '',
            regex=True
        )

        # 공백 정규화
        df['reviewText'] = df['reviewText'].str.replace(
            '\s+',
            ' ',
            regex=True
        ).str.strip()

        # 짧은 리뷰 필터링 (10자 이하)
        original_count = len(df)
        df = df[df['reviewText'].str.len() > 10]
        filtered_count = original_count - len(df)

        if filtered_count > 0:
            self.logger.info(
                f"Filtered {filtered_count} short reviews (< 10 chars)"
            )

        return df

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        추가 특징 생성

        Args:
            df: 원본 데이터프레임

        Returns:
            특징이 추가된 데이터프레임
        """
        df = df.copy()

        # 리뷰 길이
        df['review_length'] = df['reviewText'].str.len()

        # 도움됨 비율 (helpful 컬럼이 있는 경우)
        if 'helpful' in df.columns:
            df['helpful_ratio'] = df['helpful'].apply(
                lambda x: x[0] / x[1] if isinstance(x, list) and len(x) == 2 and x[1] > 0 else 0
            )
        else:
            df['helpful_ratio'] = 0.0

        # 시간 변환 (unixReviewTime이 있는 경우)
        if 'unixReviewTime' in df.columns:
            df['date'] = pd.to_datetime(df['unixReviewTime'], unit='s')
        elif 'reviewTime' in df.columns:
            df['date'] = pd.to_datetime(df['reviewTime'])
        else:
            df['date'] = pd.Timestamp.now()

        # 감성 라벨 (별점 기준)
        if 'overall' in df.columns:
            df['sentiment_label'] = df['overall'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
            )
        else:
            df['sentiment_label'] = 'neutral'

        self.logger.info(
            "Added features",
            features=['review_length', 'helpful_ratio', 'date', 'sentiment_label']
        )

        return df

    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """
        데이터프레임 검증

        Args:
            df: 검증할 데이터프레임

        Returns:
            검증 성공 여부

        Raises:
            DataValidationError: 검증 실패 시
        """
        required_columns = ['reviewText', 'overall']

        # 필수 컬럼 확인
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        # 결측치 확인
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(
                f"Found null values:\n{null_counts[null_counts > 0]}"
            )
            # 결측치 제거
            df.dropna(subset=required_columns, inplace=True)

        # 비정상적인 평점 확인
        if 'overall' in df.columns:
            invalid_ratings = df[(df['overall'] < 1) | (df['overall'] > 5)]
            if len(invalid_ratings) > 0:
                self.logger.warning(
                    f"Found {len(invalid_ratings)} invalid ratings. Removing..."
                )

        return True

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        전체 전처리 파이프라인 실행

        Args:
            df: 원본 데이터프레임

        Returns:
            전처리된 데이터프레임
        """
        self.logger.info(f"Starting preprocessing for {len(df)} reviews")

        # 검증
        self.validate_dataframe(df)

        # 텍스트 정제
        df = self.clean_text(df)

        # 특징 추가
        df = self.add_features(df)

        self.logger.info(
            f"Preprocessing complete. Final count: {len(df)} reviews",
            positive=int((df['sentiment_label'] == 'positive').sum()),
            negative=int((df['sentiment_label'] == 'negative').sum()),
            neutral=int((df['sentiment_label'] == 'neutral').sum())
        )

        return df

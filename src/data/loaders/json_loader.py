"""
JSON 파일에서 리뷰 데이터 로드
"""
import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict
from utils.logger import get_logger
from core.exceptions import DataLoadError

logger = get_logger(__name__)


class JSONReviewLoader:
    """JSON 형식의 리뷰 데이터 로더"""

    def __init__(self, filepath: str):
        """
        Args:
            filepath: JSON 파일 경로
        """
        self.filepath = Path(filepath)
        self.logger = logger

        if not self.filepath.exists():
            raise DataLoadError(f"File not found: {filepath}")

    def load(
        self,
        product_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        JSON 파일에서 리뷰 로드

        Args:
            product_id: 특정 제품 ID (ASIN)로 필터링 (선택)
            limit: 로드할 최대 리뷰 수 (선택)

        Returns:
            리뷰 데이터프레임

        Raises:
            DataLoadError: 데이터 로드 실패 시
        """
        reviews: List[Dict] = []

        try:
            self.logger.info(
                f"Loading reviews from {self.filepath}",
                product_id=product_id,
                limit=limit
            )

            with open(self.filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    # limit 체크
                    if limit and len(reviews) >= limit:
                        break

                    try:
                        review = json.loads(line.strip())

                        # product_id 필터링
                        if product_id and review.get('asin') != product_id:
                            continue

                        reviews.append(review)

                    except json.JSONDecodeError as e:
                        self.logger.warning(
                            f"Failed to parse line {i + 1}",
                            error=str(e)
                        )
                        continue

            if not reviews:
                raise DataLoadError(
                    f"No reviews found"
                    f"{' for product ' + product_id if product_id else ''}"
                )

            df = pd.DataFrame(reviews)

            self.logger.info(
                f"Successfully loaded {len(df)} reviews",
                columns=list(df.columns)
            )

            return df

        except FileNotFoundError:
            raise DataLoadError(f"File not found: {self.filepath}")
        except Exception as e:
            raise DataLoadError(f"Failed to load data: {str(e)}")

    def load_sample(self, n: int = 100) -> pd.DataFrame:
        """
        샘플 데이터 로드 (빠른 테스트용)

        Args:
            n: 샘플 개수

        Returns:
            샘플 데이터프레임
        """
        return self.load(limit=n)

"""
인사이트 추출 에이전트 (Insight Extraction Agent)

Phase 2: LLM 기반 부정 리뷰 분석 및 인사이트 도출

주요 기능:
- 부정 리뷰(1-2점)에서 고객 불만 사항 추출
- 제품 측면(aspect) 별 문제점 분류
- 빈도/심각도 기반 우선순위 분석
- LLM 활용 (gpt-oss:20b)
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd
from jinja2 import Template

from core.base_agent import BaseAgent
from core.exceptions import AgentExecutionError, LLMAPIError
from services.llm_service import LLMService
from utils.error_handler import retry_on_error, log_execution_time


class InsightExtractionAgent(BaseAgent):
    """
    인사이트 추출 에이전트

    부정 리뷰를 LLM으로 분석하여 주요 pain point와 제품 측면별 문제점을 추출
    """

    VERSION = "2.0.0"

    def __init__(
        self,
        config: Dict[str, Any],
        llm_service: LLMService,
        logger=None
    ):
        """
        Args:
            config: 설정 딕셔너리
            llm_service: LLM 서비스 인스턴스
            logger: 로거 (선택)
        """
        super().__init__(config, logger)
        self.llm_service = llm_service

        # 프롬프트 템플릿 로드
        template_path = Path(__file__).parent.parent / "prompts" / "insight_extraction.jinja2"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = Template(f.read())

        self.logger.info(f"InsightExtractionAgent initialized (v{self.VERSION})")

    @log_execution_time
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        인사이트 추출 실행 (긍정 + 부정 리뷰 모두 분석)

        Args:
            input_data: {
                "dataframe": pd.DataFrame,  # 전체 리뷰 데이터
                "stats": Dict,              # 기본 통계
                "negative_reviews": pd.DataFrame,  # 부정 리뷰
                "positive_reviews": pd.DataFrame   # 긍정 리뷰 (신규)
            }

        Returns:
            {
                "insights": Dict,  # 추출된 인사이트
                "pain_points": List[Dict],  # 부정 리뷰에서 도출
                "strengths": List[Dict],    # 긍정 리뷰에서 도출 (신규)
                "product_aspects": Dict,
                "success": bool,
                "error": Optional[str]
            }
        """
        try:
            self.logger.info("Starting insight extraction (positive + negative)...")

            # 입력 데이터 검증
            self._validate_input(input_data)

            df = input_data["dataframe"]
            stats = input_data["stats"]
            negative_df = input_data.get("negative_reviews", pd.DataFrame())
            positive_df = input_data.get("positive_reviews", pd.DataFrame())

            # 부정과 긍정 리뷰 모두 없으면 조기 종료
            if len(negative_df) == 0 and len(positive_df) == 0:
                self.logger.warning("No reviews to analyze")
                return {
                    "insights": {},
                    "pain_points": [],
                    "strengths": [],
                    "product_aspects": {},
                    "success": True,
                    "error": None,
                    "message": "No reviews found for analysis"
                }

            # 배치 처리 여부 결정
            total_reviews = len(negative_df) + len(positive_df)
            BATCH_SIZE = 100  # 배치당 최대 리뷰 수

            if total_reviews <= BATCH_SIZE:
                # 100개 이하: 일반 처리
                self.logger.info(f"Processing {total_reviews} reviews in single batch")
                prompt = self._prepare_prompt(df, stats, negative_df, positive_df)
                insights_data = self._call_llm(prompt)
                insights = self._process_insights(insights_data)
            else:
                # 100개 초과: 배치 처리
                self.logger.info(f"Processing {total_reviews} reviews in batches of {BATCH_SIZE}")
                insights = self._process_in_batches(df, stats, negative_df, positive_df, BATCH_SIZE)

            # 메트릭 기록
            self.metrics["pain_points_count"] = len(insights.get("pain_points", []))
            self.metrics["strengths_count"] = len(insights.get("strengths", []))
            self.metrics["aspects_analyzed"] = len(insights.get("product_aspects", {}))

            self.logger.info(
                f"Insight extraction completed: "
                f"{self.metrics['pain_points_count']} pain points, "
                f"{self.metrics['strengths_count']} strengths, "
                f"{self.metrics['aspects_analyzed']} aspects"
            )

            return {
                "insights": insights,
                "pain_points": insights.get("pain_points", []),
                "strengths": insights.get("strengths", []),
                "product_aspects": insights.get("product_aspects", {}),
                "summary": insights.get("summary", ""),
                "success": True,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"Insight extraction failed: {str(e)}", exc_info=True)
            raise AgentExecutionError(f"InsightAgent failed: {str(e)}")

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """입력 데이터 검증"""
        required_keys = ["dataframe", "stats", "negative_reviews"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Missing required key: {key}")

        if not isinstance(input_data["dataframe"], pd.DataFrame):
            raise TypeError("'dataframe' must be a pandas DataFrame")

        if not isinstance(input_data["negative_reviews"], pd.DataFrame):
            raise TypeError("'negative_reviews' must be a pandas DataFrame")

    def _prepare_prompt(
        self,
        df: pd.DataFrame,
        stats: Dict,
        negative_df: pd.DataFrame,
        positive_df: pd.DataFrame
    ) -> str:
        """
        Jinja2 템플릿을 사용하여 프롬프트 생성 (긍정 + 부정)

        스마트 샘플링 제거: 전체 리뷰를 무작위로 섞어서 모두 LLM에 전달
        """
        # 부정 리뷰 (전체, 무작위 섞기)
        negative_count = len(negative_df)
        negative_samples = self._sample_reviews(negative_df, negative_count)

        # 긍정 리뷰 (전체, 무작위 섞기)
        positive_count = len(positive_df)
        positive_samples = self._sample_reviews(positive_df, positive_count)

        self.logger.info(
            f"LLM analysis: "
            f"{negative_count} negative, "
            f"{positive_count} positive (total: {negative_count + positive_count})"
        )

        # 템플릿 렌더링
        prompt = self.prompt_template.render(
            total_reviews=len(df),
            negative_count=negative_count,
            positive_count=positive_count,
            avg_rating=stats.get("avg_rating", 0),
            negative_sample_size=negative_count,
            positive_sample_size=positive_count,
            negative_reviews=negative_samples,
            positive_reviews=positive_samples
        )

        self.logger.debug(
            f"Prompt prepared: {len(prompt)} characters, "
            f"{negative_count + positive_count} total reviews"
        )
        return prompt

    def _sample_reviews(self, df: pd.DataFrame, sample_size: int) -> List[Dict]:
        """리뷰 샘플링 (완전 무작위 섞기)"""
        if len(df) == 0 or sample_size == 0:
            return []

        if sample_size >= len(df):
            # 전체를 무작위로 섞어서 반환
            return df.sample(frac=1.0).to_dict('records')

        # 지정된 개수만큼 무작위 샘플링
        sample_df = df.sample(n=min(sample_size, len(df)))

        return sample_df.to_dict('records')

    @retry_on_error(max_retries=3, delay=2)
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        LLM 호출 (retry 포함)

        gpt-oss:20b의 thinking 기능을 활용하여 분석
        대규모 리뷰 처리를 위해 max_tokens 동적 조정
        """
        try:
            self.logger.info("Calling LLM for insight extraction...")

            # 대규모 분석을 위해 max_tokens 증가
            max_tokens = self.config.get("max_tokens", 3000)  # 2000 → 3000

            # JSON 모드로 호출
            insights = self.llm_service.generate_json(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.config.get("temperature", 0.7),
                retries=5  # 재시도 횟수 증가 (3 → 5)
            )

            if insights is None:
                raise LLMAPIError("LLM returned None response after all retries")

            self.logger.info("LLM call successful")
            return insights

        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise LLMAPIError(f"Failed to call LLM: {str(e)}")

    def _process_insights(self, insights_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 응답 후처리 및 검증 (pain_points + strengths)
        """
        # 필수 키 확인
        if "pain_points" not in insights_data:
            self.logger.warning("Missing 'pain_points' in LLM response")
            insights_data["pain_points"] = []

        if "strengths" not in insights_data:
            self.logger.warning("Missing 'strengths' in LLM response")
            insights_data["strengths"] = []

        if "product_aspects" not in insights_data:
            self.logger.warning("Missing 'product_aspects' in LLM response")
            insights_data["product_aspects"] = {}

        # pain_points 검증
        for i, pain_point in enumerate(insights_data.get("pain_points", [])):
            required_fields = ["issue", "frequency", "severity"]
            for field in required_fields:
                if field not in pain_point:
                    self.logger.warning(
                        f"Pain point #{i+1} missing '{field}', adding default"
                    )
                    pain_point[field] = "unknown"

        # strengths 검증
        for i, strength in enumerate(insights_data.get("strengths", [])):
            required_fields = ["feature", "frequency"]
            for field in required_fields:
                if field not in strength:
                    self.logger.warning(
                        f"Strength #{i+1} missing '{field}', adding default"
                    )
                    strength[field] = "unknown"

        return insights_data

    def get_top_pain_points(
        self,
        insights: Dict[str, Any],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        상위 N개의 pain point 반환 (심각도 기준)

        Args:
            insights: execute() 결과의 insights
            top_n: 반환할 개수

        Returns:
            상위 pain point 리스트
        """
        pain_points = insights.get("pain_points", [])

        # 심각도 순위
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}

        sorted_points = sorted(
            pain_points,
            key=lambda x: severity_order.get(x.get("severity", "low"), 0),
            reverse=True
        )

        return sorted_points[:top_n]

    def _process_in_batches(
        self,
        df: pd.DataFrame,
        stats: Dict,
        negative_df: pd.DataFrame,
        positive_df: pd.DataFrame,
        batch_size: int
    ) -> Dict[str, Any]:
        """
        대량 리뷰를 배치로 나누어 처리하고 결과 통합

        Args:
            df: 전체 DataFrame
            stats: 기본 통계
            negative_df: 부정 리뷰 DataFrame
            positive_df: 긍정 리뷰 DataFrame
            batch_size: 배치당 최대 리뷰 수

        Returns:
            통합된 인사이트
        """
        import math

        total_negative = len(negative_df)
        total_positive = len(positive_df)
        total_reviews = total_negative + total_positive

        # 배치 수 계산
        num_batches = math.ceil(total_reviews / batch_size)

        # 긍정/부정 비율 유지하면서 배치 크기 계산
        negative_ratio = total_negative / total_reviews if total_reviews > 0 else 0.5
        positive_ratio = total_positive / total_reviews if total_reviews > 0 else 0.5

        self.logger.info(
            f"Batch processing: {num_batches} batches "
            f"(negative ratio: {negative_ratio:.2%}, positive ratio: {positive_ratio:.2%})"
        )

        batch_results = []

        for i in range(num_batches):
            # 각 배치의 크기 계산
            neg_batch_size = int(batch_size * negative_ratio)
            pos_batch_size = int(batch_size * positive_ratio)

            # 배치 인덱스 계산
            neg_start = i * neg_batch_size
            neg_end = min((i + 1) * neg_batch_size, total_negative)
            pos_start = i * pos_batch_size
            pos_end = min((i + 1) * pos_batch_size, total_positive)

            # 배치 추출
            neg_batch = negative_df.iloc[neg_start:neg_end] if neg_end > neg_start else pd.DataFrame()
            pos_batch = positive_df.iloc[pos_start:pos_end] if pos_end > pos_start else pd.DataFrame()

            batch_total = len(neg_batch) + len(pos_batch)
            if batch_total == 0:
                continue

            self.logger.info(
                f"Batch {i+1}/{num_batches}: "
                f"{len(neg_batch)} negative + {len(pos_batch)} positive = {batch_total} reviews"
            )

            # 배치 처리
            try:
                prompt = self._prepare_prompt(df, stats, neg_batch, pos_batch)
                insights_data = self._call_llm(prompt)
                insights = self._process_insights(insights_data)
                batch_results.append(insights)

                self.logger.info(f"Batch {i+1}/{num_batches} completed successfully")

            except Exception as e:
                self.logger.error(f"Batch {i+1}/{num_batches} failed: {str(e)}")
                # 배치 실패 시 계속 진행
                continue

        # 배치 결과 통합
        if not batch_results:
            self.logger.error("All batches failed!")
            return {
                "pain_points": [],
                "strengths": [],
                "product_aspects": {},
                "summary": "Batch processing failed"
            }

        merged_insights = self._merge_batch_results(batch_results)
        self.logger.info(
            f"Batch processing complete. Merged results: "
            f"{len(merged_insights.get('pain_points', []))} pain points, "
            f"{len(merged_insights.get('strengths', []))} strengths"
        )

        return merged_insights

    def _merge_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        여러 배치의 결과를 통합

        Args:
            batch_results: 배치별 인사이트 리스트

        Returns:
            통합된 인사이트
        """
        all_pain_points = []
        all_strengths = []
        all_aspects = {}

        for batch in batch_results:
            # Pain points 수집
            all_pain_points.extend(batch.get("pain_points", []))

            # Strengths 수집
            all_strengths.extend(batch.get("strengths", []))

            # Product aspects 병합 (mention_count 합산)
            batch_aspects = batch.get("product_aspects", {})
            for aspect, data in batch_aspects.items():
                if aspect not in all_aspects:
                    all_aspects[aspect] = data.copy()
                else:
                    # mention_count 합산
                    all_aspects[aspect]["mention_count"] += data.get("mention_count", 0)

        # Pain points 중복 제거 및 빈도 기준 정렬 (상위 10개)
        unique_pain_points = self._deduplicate_and_rank(all_pain_points, "issue", limit=10)

        # Strengths 중복 제거 및 빈도 기준 정렬 (상위 10개)
        unique_strengths = self._deduplicate_and_rank(all_strengths, "feature", limit=10)

        return {
            "pain_points": unique_pain_points,
            "strengths": unique_strengths,
            "product_aspects": all_aspects,
            "summary": f"Merged insights from {len(batch_results)} batches"
        }

    def _deduplicate_and_rank(
        self,
        items: List[Dict[str, Any]],
        key_field: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        아이템 중복 제거 및 순위 매기기

        Args:
            items: 아이템 리스트
            key_field: 중복 판단 기준 필드 (e.g., "issue", "feature")
            limit: 반환할 최대 개수

        Returns:
            중복 제거 및 정렬된 아이템 리스트
        """
        from collections import defaultdict

        # 텍스트 유사도 기반 그룹화 (간단한 버전: 소문자화 + 공백 제거)
        grouped = defaultdict(list)

        for item in items:
            key = item.get(key_field, "").lower().strip()
            if key:
                grouped[key].append(item)

        # 각 그룹에서 대표 아이템 선택 (첫 번째)
        unique_items = []
        for key, group in grouped.items():
            representative = group[0]
            # frequency를 숫자로 파싱 시도
            freq_str = representative.get("frequency", "0")
            try:
                # "10회", "20%" 등에서 숫자 추출
                freq_num = int(''.join(filter(str.isdigit, str(freq_str))) or 0)
            except:
                freq_num = len(group)  # 파싱 실패 시 그룹 크기 사용

            representative["_freq_num"] = freq_num
            unique_items.append(representative)

        # 빈도 기준 정렬
        unique_items.sort(key=lambda x: x.get("_freq_num", 0), reverse=True)

        # _freq_num 필드 제거
        for item in unique_items:
            item.pop("_freq_num", None)

        return unique_items[:limit]

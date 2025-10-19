"""
경쟁사 비교 분석 에이전트 (Phase 3)

여러 제품의 리뷰를 병렬로 분석하여 경쟁 우위를 파악합니다.
"""

from typing import Dict, Any, List
from pathlib import Path
from core.base_agent import BaseAgent
from core.orchestrator import ReviewAnalysisOrchestrator
import concurrent.futures


class CompetitorAnalysisAgent(BaseAgent):
    """
    경쟁사 비교 분석 에이전트 (Phase 3)

    기능:
    - 여러 제품 리뷰 병렬 분석
    - 제품 간 평점/감정/aspect 비교
    - 경쟁 우위/열위 분석
    """

    VERSION = "3.0.0"

    def __init__(self, config: Dict[str, Any], logger=None):
        super().__init__(config, logger)
        self.orchestrator_config = config

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        경쟁사 비교 분석 실행

        Args:
            input_data: {
                'products': [
                    {'product_id': 'ASIN1', 'data_path': 'path1.json'},
                    {'product_id': 'ASIN2', 'data_path': 'path2.json'},
                    ...
                ],
                'limit': int  # 제품당 분석할 리뷰 수
            }

        Returns:
            경쟁사 비교 분석 결과
        """
        self._log_execution_start()

        products = input_data.get('products', [])
        limit = input_data.get('limit', 100)

        if len(products) < 2:
            self.logger.warning("Need at least 2 products for comparison")
            return {"error": "Insufficient products for comparison"}

        # 병렬로 각 제품 분석
        product_results = self._analyze_products_parallel(products, limit)

        # 비교 분석
        comparison = self._compare_products(product_results)

        result = {
            "product_count": len(products),
            "product_results": product_results,
            "comparison": comparison
        }

        self._log_execution_end()

        return result

    def _analyze_products_parallel(
        self,
        products: List[Dict[str, str]],
        limit: int
    ) -> Dict[str, Dict[str, Any]]:
        """
        여러 제품을 병렬로 분석

        Args:
            products: 제품 정보 리스트
            limit: 제품당 리뷰 수

        Returns:
            {product_id: analysis_result}
        """
        self.logger.info(f"Analyzing {len(products)} products in parallel...")

        results = {}

        # 병렬 처리 (ThreadPoolExecutor 사용)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # 각 제품에 대한 분석 작업 제출
            future_to_product = {
                executor.submit(
                    self._analyze_single_product,
                    product['product_id'],
                    product['data_path'],
                    limit
                ): product['product_id']
                for product in products
            }

            # 결과 수집
            for future in concurrent.futures.as_completed(future_to_product):
                product_id = future_to_product[future]
                try:
                    result = future.result()
                    results[product_id] = result
                    self.logger.info(f"Product {product_id} analyzed successfully")
                except Exception as e:
                    self.logger.error(f"Product {product_id} analysis failed: {e}")
                    results[product_id] = {"error": str(e)}

        return results

    def _analyze_single_product(
        self,
        product_id: str,
        data_path: str,
        limit: int
    ) -> Dict[str, Any]:
        """
        단일 제품 분석 (Orchestrator 사용)

        Args:
            product_id: 제품 ID
            data_path: 데이터 파일 경로
            limit: 리뷰 수

        Returns:
            분석 결과
        """
        orchestrator = ReviewAnalysisOrchestrator(self.orchestrator_config)

        result = orchestrator.run_analysis(
            data_path=data_path,
            product_id=product_id,
            limit=limit,
            enable_llm=False  # 빠른 분석을 위해 LLM 비활성화
        )

        # 핵심 메트릭만 추출
        return {
            "product_id": product_id,
            "avg_rating": result['basic_stats'].get('avg_rating', 0),
            "total_reviews": result['basic_stats'].get('total_reviews', 0),
            "sentiment_distribution": result['sentiment_analysis'].get('sentiment_distribution', {}),
            "absa_summary": result['sentiment_analysis'].get('aspect_summary', [])[:5]  # 상위 5개만
        }

    def _compare_products(
        self,
        product_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        제품 간 비교 분석

        Args:
            product_results: 제품별 분석 결과

        Returns:
            비교 분석 결과
        """
        # 평점 순위
        rating_ranking = sorted(
            [(pid, data.get('avg_rating', 0)) for pid, data in product_results.items()],
            key=lambda x: x[1],
            reverse=True
        )

        # 긍정 리뷰 비율 순위
        sentiment_ranking = []
        for pid, data in product_results.items():
            dist = data.get('sentiment_distribution', {})
            total = sum(dist.values()) or 1
            positive_ratio = dist.get('positive', 0) / total * 100
            sentiment_ranking.append((pid, positive_ratio))

        sentiment_ranking.sort(key=lambda x: x[1], reverse=True)

        # 강점/약점 aspect 추출
        aspect_comparison = self._compare_aspects(product_results)

        return {
            "rating_ranking": [
                {"product_id": pid, "avg_rating": rating}
                for pid, rating in rating_ranking
            ],
            "sentiment_ranking": [
                {"product_id": pid, "positive_ratio": ratio}
                for pid, ratio in sentiment_ranking
            ],
            "aspect_comparison": aspect_comparison,
            "winner": rating_ranking[0][0] if rating_ranking else None
        }

    def _compare_aspects(
        self,
        product_results: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Aspect 기반 제품 비교

        Args:
            product_results: 제품별 분석 결과

        Returns:
            Aspect 비교 결과
        """
        # 각 제품의 상위 aspect 수집
        aspect_scores = {}

        for pid, data in product_results.items():
            absa_summary = data.get('absa_summary', [])
            for aspect_data in absa_summary:
                aspect_name = aspect_data.get('aspect')
                avg_rating = aspect_data.get('avg_rating', 0)

                if aspect_name not in aspect_scores:
                    aspect_scores[aspect_name] = {}

                aspect_scores[aspect_name][pid] = avg_rating

        # Aspect별 최고 제품 찾기
        aspect_comparison = []
        for aspect_name, scores in aspect_scores.items():
            best_product = max(scores.items(), key=lambda x: x[1])
            aspect_comparison.append({
                "aspect": aspect_name,
                "best_product": best_product[0],
                "best_score": best_product[1],
                "scores": scores
            })

        return aspect_comparison

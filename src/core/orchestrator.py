"""
전체 워크플로우를 관리하는 오케스트레이터 (Phase 3 버전)

Phase 2 기능:
- LLM 기반 인사이트 추출 (InsightAgent)
- 실행 계획 수립 (ActionPlanningAgent)
- 최종 리포트 생성 (ReportAgent)

Phase 3 추가 기능:
- 시각화 서비스 (VisualizationService)
"""
import time
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

from data.loaders.json_loader import JSONReviewLoader
from data.preprocessor import DataPreprocessor
from agents.data_collection_agent import DataCollectionAgent
from agents.sentiment_agent import SentimentAnalysisAgent
from agents.insight_agent import InsightExtractionAgent
from agents.action_planning_agent import ActionPlanningAgent
from agents.report_agent import ReportGenerationAgent
from services.llm_service import create_llm_service
from services.visualization_service import VisualizationService
from utils.logger import get_logger
from core.exceptions import ReviewAnalysisException


class ReviewAnalysisOrchestrator:
    """리뷰 분석 파이프라인 오케스트레이터 (Phase 3)"""

    VERSION = "3.0.0"

    def __init__(self, config: Dict[str, Any], llm_config_path: Optional[str] = None):
        """
        Args:
            config: 설정 딕셔너리
            llm_config_path: LLM 설정 파일 경로 (선택, 기본: config/llm_config.yaml)
        """
        self.config = config
        self.logger = get_logger(
            "Orchestrator",
            level=config.get('log_level', 'INFO')
        )

        # LLM 서비스 초기화
        if llm_config_path is None:
            llm_config_path = Path(__file__).parent.parent / "config" / "llm_config.yaml"

        self.llm_service = self._init_llm_service(llm_config_path)

        # Phase 1 에이전트
        self.data_agent = DataCollectionAgent(config)
        self.sentiment_agent = SentimentAnalysisAgent(config)

        # Phase 2 에이전트 (LLM 기반)
        self.insight_agent = InsightExtractionAgent(config, self.llm_service)
        self.action_planning_agent = ActionPlanningAgent(config, self.llm_service)
        self.report_agent = ReportGenerationAgent(config, self.llm_service)

        # Phase 3 서비스
        self.visualization_service = VisualizationService(logger=self.logger)

        # 데이터 전처리기
        self.preprocessor = DataPreprocessor()

        self.results: Dict[str, Any] = {}

    def _init_llm_service(self, config_path: Path):
        """LLM 서비스 초기화"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                llm_config = yaml.safe_load(f)

            service = create_llm_service(llm_config)
            self.logger.info(f"LLM service initialized: {llm_config.get('provider', 'unknown')}")
            return service

        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM service: {e}")
            self.logger.warning("LLM-based features will be disabled")
            return None

    def run_analysis(
        self,
        data_path: str,
        product_id: Optional[str] = None,
        limit: Optional[int] = None,
        enable_llm: bool = True
    ) -> Dict[str, Any]:
        """
        전체 분석 파이프라인 실행 (Phase 2)

        Args:
            data_path: 데이터 파일 경로
            product_id: 제품 ID (선택)
            limit: 로드할 리뷰 수 제한 (선택)
            enable_llm: LLM 기반 분석 활성화 (기본: True)

        Returns:
            분석 결과 딕셔너리
        """
        self.logger.info("=" * 60)
        self.logger.info("🚀 Review Analysis System Started (Phase 2)")
        self.logger.info("=" * 60)

        start_time = time.time()

        try:
            # Stage 1: 데이터 로드
            self.logger.info("\n📥 Stage 1: Data Loading...")
            df = self._load_data(data_path, product_id, limit)

            # Stage 2: 데이터 전처리
            self.logger.info("\n🔧 Stage 2: Data Preprocessing...")
            df = self._preprocess_data(df)

            # Stage 3: 데이터 수집 및 기본 통계
            self.logger.info("\n📊 Stage 3: Data Collection & Statistics...")
            collection_result = self._collect_data(df)

            # Stage 4: 감성 분석
            self.logger.info("\n💭 Stage 4: Sentiment Analysis...")
            sentiment_result = self._analyze_sentiment(collection_result)

            # 결과 통합 (Phase 1)
            self.results = {
                "basic_stats": collection_result['stats'],
                "sentiment_analysis": sentiment_result,
                "negative_reviews_count": len(collection_result['negative_reviews']),
                "recent_reviews_count": len(collection_result['recent_reviews']),
                "metadata": {
                    "product_id": product_id,
                    "total_reviews_analyzed": collection_result['stats']['total_reviews'],
                    "version": self.VERSION,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }

            # Phase 2: LLM 기반 분석 (선택적)
            if enable_llm and self.llm_service is not None:
                # Stage 5: 인사이트 추출
                self.logger.info("\n🔍 Stage 5: Insight Extraction (LLM)...")
                insight_result = self._extract_insights(collection_result)
                self.results["insights"] = insight_result

                # Stage 6: 실행 계획 수립
                self.logger.info("\n📋 Stage 6: Action Planning (LLM)...")
                action_result = self._plan_actions(
                    insight_result.get("pain_points", []),
                    collection_result['stats']
                )
                self.results["action_plan"] = action_result

                # Stage 7: 최종 리포트 생성
                self.logger.info("\n📄 Stage 7: Report Generation (LLM)...")
                report_result = self._generate_report(
                    product_id,
                    collection_result['stats'],
                    insight_result.get("insights", {}),
                    action_result.get("action_plan", {})
                )
                self.results["final_report"] = report_result

                self.logger.info("\n✅ LLM-based analysis completed")
            else:
                if not enable_llm:
                    self.logger.info("\n⏩ LLM analysis skipped (enable_llm=False)")
                else:
                    self.logger.warning("\n⚠️  LLM service not available, skipping Phase 2")

            # Phase 3: Stage 8 - 시각화
            self.logger.info("\n📊 Stage 8: Visualization (Phase 3)...")
            charts = self._generate_visualizations(df, sentiment_result)
            if charts:
                self.results["visualizations"] = {
                    "charts": {name: str(path) for name, path in charts.items()},
                    "chart_count": len(charts)
                }
                self.logger.info(f"Generated {len(charts)} charts")

            # 성능 메트릭
            duration = time.time() - start_time
            self.logger.info(f"\n⏱️  Total processing time: {duration:.2f}s")
            self.logger.info(f"📈 Processing speed: {collection_result['stats']['total_reviews'] / duration:.2f} reviews/s")

            self.logger.info("\n✅ Analysis Complete!")
            self.logger.info("=" * 60)

            return self.results

        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise ReviewAnalysisException(f"Pipeline execution failed: {str(e)}")

    def _load_data(
        self,
        data_path: str,
        product_id: Optional[str],
        limit: Optional[int]
    ):
        """데이터 로드"""
        loader = JSONReviewLoader(data_path)
        df = loader.load(product_id=product_id, limit=limit)

        self.logger.info(
            f"Loaded {len(df)} reviews",
            product_id=product_id or "all"
        )

        return df

    def _preprocess_data(self, df):
        """데이터 전처리"""
        df = self.preprocessor.process(df)

        self.logger.info(
            f"Preprocessing complete",
            final_count=len(df)
        )

        return df

    def _collect_data(self, df):
        """데이터 수집 및 기본 통계"""
        result = self.data_agent.execute(df)

        self.logger.info(
            "Data collection complete",
            avg_rating=round(result['stats']['avg_rating'], 2),
            negative_count=len(result['negative_reviews'])
        )

        return result

    def _analyze_sentiment(self, collection_result):
        """감성 분석"""
        result = self.sentiment_agent.execute(collection_result)

        self.logger.info(
            "Sentiment analysis complete",
            positive=result['sentiment_distribution'].get('positive', 0),
            negative=result['sentiment_distribution'].get('negative', 0)
        )

        return result

    def _extract_insights(self, collection_result):
        """인사이트 추출 (Phase 2 - 긍정/부정 모두)"""
        result = self.insight_agent.execute(collection_result)

        pain_points_count = len(result.get('pain_points', []))
        strengths_count = len(result.get('strengths', []))
        self.logger.info(
            "Insight extraction complete",
            pain_points=pain_points_count,
            strengths=strengths_count,
            aspects=len(result.get('product_aspects', {}))
        )

        return result

    def _plan_actions(self, pain_points, stats):
        """실행 계획 수립 (Phase 2)"""
        input_data = {
            "pain_points": pain_points,
            "stats": stats
        }

        result = self.action_planning_agent.execute(input_data)

        self.logger.info(
            "Action planning complete",
            quick_wins=len(result.get('quick_wins', [])),
            medium_term=len(result.get('medium_term_actions', [])),
            long_term=len(result.get('long_term_actions', []))
        )

        return result

    def _generate_report(self, product_id, stats, insights, action_plan):
        """최종 리포트 생성 (Phase 2)"""
        input_data = {
            "product_id": product_id,
            "stats": stats,
            "insights": insights,
            "action_plan": action_plan
        }

        result = self.report_agent.execute(input_data)

        self.logger.info(
            "Report generation complete",
            findings=len(result.get('key_findings', [])),
            actions=len(result.get('immediate_actions', []))
        )

        return result

    def get_summary(self) -> str:
        """
        분석 결과 요약 텍스트 생성 (Phase 2 포함)

        Returns:
            요약 텍스트
        """
        if not self.results:
            return "No analysis results available."

        stats = self.results.get('basic_stats', {})
        sentiment = self.results.get('sentiment_analysis', {})
        sentiment_dist = sentiment.get('sentiment_distribution', {})

        summary_lines = [
            "📊 리뷰 분석 결과 요약",
            "",
            "기본 통계:",
            f"- 총 리뷰 수: {stats.get('total_reviews', 0):,}개",
            f"- 평균 평점: {stats.get('avg_rating', 0):.2f}/5.0",
            f"- 평균 리뷰 길이: {stats.get('avg_review_length', 0):.0f}자",
            "",
            "감성 분석:",
            f"- 긍정 리뷰: {sentiment_dist.get('positive', 0):,}개 ({sentiment_dist.get('positive', 0) / stats.get('total_reviews', 1) * 100:.1f}%)",
            f"- 부정 리뷰: {sentiment_dist.get('negative', 0):,}개 ({sentiment_dist.get('negative', 0) / stats.get('total_reviews', 1) * 100:.1f}%)",
            f"- 중립 리뷰: {sentiment_dist.get('neutral', 0):,}개",
        ]

        # Phase 2 결과 추가
        if "insights" in self.results:
            insights = self.results["insights"]
            pain_points = insights.get("pain_points", [])

            summary_lines.extend([
                "",
                "🔍 주요 인사이트:",
                f"- 발견된 문제점: {len(pain_points)}개"
            ])

            # Top 3 pain points 표시
            for i, pain_point in enumerate(pain_points[:3], 1):
                issue = pain_point.get("issue", "Unknown")
                severity = pain_point.get("severity", "unknown")
                summary_lines.append(f"  {i}. [{severity.upper()}] {issue}")

        if "action_plan" in self.results:
            action_plan = self.results["action_plan"]
            quick_wins = action_plan.get("quick_wins", [])
            total_actions = (
                len(quick_wins) +
                len(action_plan.get("medium_term_actions", [])) +
                len(action_plan.get("long_term_actions", []))
            )

            summary_lines.extend([
                "",
                "📋 실행 계획:",
                f"- 총 권장 액션: {total_actions}개",
                f"- Quick Win: {len(quick_wins)}개"
            ])

        if "final_report" in self.results:
            report = self.results["final_report"]
            summary_lines.extend([
                "",
                "📄 최종 리포트:",
                f"- 핵심 발견: {len(report.get('key_findings', []))}개",
                f"- 즉시 조치 필요: {len(report.get('immediate_actions', []))}개"
            ])

        summary_lines.extend([
            "",
            "분석 정보:",
            f"- 제품 ID: {self.results['metadata'].get('product_id', 'N/A')}",
            f"- 분석 시각: {self.results['metadata'].get('timestamp', 'N/A')}",
            f"- 시스템 버전: {self.results['metadata'].get('version', 'N/A')}"
        ])

        return "\n".join(summary_lines)

    def save_results(self, output_path: str):
        """
        결과를 JSON 파일로 저장

        Args:
            output_path: 출력 파일 경로
        """
        import json

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to {output_path}")

    def _generate_visualizations(self, df, sentiment_result):
        """
        시각화 생성 (Phase 3)

        Args:
            df: 리뷰 DataFrame
            sentiment_result: 감성 분석 결과

        Returns:
            생성된 차트 경로 딕셔너리
        """
        try:
            sentiment_dist = sentiment_result.get('sentiment_distribution', {})
            absa_results = sentiment_result.get('absa', None)

            charts = self.visualization_service.generate_all_charts(
                df=df,
                sentiment_distribution=sentiment_dist,
                absa_results=absa_results
            )

            return charts

        except Exception as e:
            self.logger.warning(f"Visualization failed: {str(e)}")
            return None

"""
리포트 생성 에이전트 (Report Generation Agent)

Phase 2: 최종 경영진 요약 리포트 생성

주요 기능:
- 전체 분석 결과를 비즈니스 친화적인 요약으로 변환
- Executive Summary (2-3 문장)
- Key Findings (데이터 기반)
- Immediate Actions (우선순위 Top 3)
- Business Impact 예측
- LLM 활용 (gpt-oss:20b)
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
from jinja2 import Template

from core.base_agent import BaseAgent
from core.exceptions import AgentExecutionError, LLMAPIError
from services.llm_service import LLMService
from utils.error_handler import retry_on_error, log_execution_time


class ReportGenerationAgent(BaseAgent):
    """
    리포트 생성 에이전트

    전체 분석 파이프라인 결과를 경영진용 요약 리포트로 변환
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
        template_path = Path(__file__).parent.parent / "prompts" / "report_generation.jinja2"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = Template(f.read())

        self.logger.info(f"ReportGenerationAgent initialized (v{self.VERSION})")

    @log_execution_time
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        최종 리포트 생성

        Args:
            input_data: {
                "product_id": str,              # 제품 ID
                "stats": Dict,                  # 기본 통계
                "insights": Dict,               # InsightAgent 결과
                "action_plan": Dict,            # ActionPlanningAgent 결과
                "sentiment_distribution": Dict  # (선택) 감정 분포
            }

        Returns:
            {
                "report": Dict,  # 전체 리포트
                "executive_summary": str,
                "key_findings": List[str],
                "immediate_actions": List[Dict],
                "business_impact": Dict,
                "success": bool,
                "error": Optional[str]
            }
        """
        try:
            self.logger.info("Starting report generation...")

            # 입력 데이터 검증
            self._validate_input(input_data)

            # 프롬프트 준비
            prompt = self._prepare_prompt(input_data)

            # LLM 호출 (JSON 모드)
            report_data = self._call_llm(prompt)

            # 결과 검증 및 후처리
            report = self._process_report(report_data)

            # 메타데이터 추가
            report["metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "product_id": input_data.get("product_id", "unknown"),
                "agent_version": self.VERSION,
                "total_reviews": input_data["stats"].get("total_reviews", 0)
            }

            # 메트릭 기록
            self.metrics["findings_count"] = len(report.get("key_findings", []))
            self.metrics["actions_count"] = len(report.get("immediate_actions", []))

            self.logger.info(
                f"Report generation completed: "
                f"{self.metrics['findings_count']} findings, "
                f"{self.metrics['actions_count']} immediate actions"
            )

            return {
                "report": report,
                "executive_summary": report.get("executive_summary", ""),
                "key_findings": report.get("key_findings", []),
                "immediate_actions": report.get("immediate_actions", []),
                "business_impact": report.get("business_impact", {}),
                "metrics_to_track": report.get("metrics_to_track", []),
                "metadata": report["metadata"],
                "success": True,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}", exc_info=True)
            raise AgentExecutionError(f"ReportAgent failed: {str(e)}")

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """입력 데이터 검증"""
        required_keys = ["stats", "insights", "action_plan"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Missing required key: {key}")

    def _prepare_prompt(self, input_data: Dict[str, Any]) -> str:
        """
        Jinja2 템플릿을 사용하여 프롬프트 생성
        """
        stats = input_data["stats"]
        insights = input_data["insights"]
        action_plan = input_data["action_plan"]

        # 감정 분포 계산
        total = stats.get("total_reviews", 0)
        positive_count = stats.get("positive_count", 0)
        neutral_count = stats.get("neutral_count", 0)
        negative_count = stats.get("negative_count", 0)

        positive_pct = round((positive_count / total * 100), 1) if total > 0 else 0
        neutral_pct = round((neutral_count / total * 100), 1) if total > 0 else 0
        negative_pct = round((negative_count / total * 100), 1) if total > 0 else 0

        # 날짜 범위 (간단히 "recent"로 표시, 실제로는 데이터에서 추출 가능)
        date_range = input_data.get("date_range", "recent reviews")

        # 템플릿 렌더링
        prompt = self.prompt_template.render(
            product_id=input_data.get("product_id", "N/A"),
            total_reviews=total,
            date_range=date_range,
            avg_rating=stats.get("avg_rating", 0),
            positive_count=positive_count,
            positive_pct=positive_pct,
            neutral_count=neutral_count,
            neutral_pct=neutral_pct,
            negative_count=negative_count,
            negative_pct=negative_pct,
            insights=insights.get("pain_points", []),
            quick_wins=action_plan.get("quick_wins", []),
            medium_term_actions=action_plan.get("medium_term_actions", [])
        )

        self.logger.debug(f"Prompt prepared: {len(prompt)} characters")
        return prompt

    @retry_on_error(max_retries=3, delay=2)
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        LLM 호출 (retry 포함)

        gpt-oss:20b의 thinking 기능을 활용하여 비즈니스 요약 생성
        """
        try:
            self.logger.info("Calling LLM for report generation...")

            # JSON 모드로 호출
            report = self.llm_service.generate_json(
                prompt=prompt,
                max_tokens=self.config.get("max_tokens", 2000),
                temperature=self.config.get("temperature", 0.7)
            )

            if report is None:
                raise LLMAPIError("LLM returned None response")

            self.logger.info("LLM call successful")
            return report

        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise LLMAPIError(f"Failed to call LLM: {str(e)}")

    def _process_report(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 응답 후처리 및 검증
        """
        # 필수 키 확인
        expected_keys = [
            "executive_summary",
            "key_findings",
            "immediate_actions",
            "business_impact",
            "metrics_to_track"
        ]

        for key in expected_keys:
            if key not in report_data:
                self.logger.warning(f"Missing '{key}' in LLM response, adding default")
                if key in ["executive_summary"]:
                    report_data[key] = "No summary available"
                elif key in ["business_impact"]:
                    report_data[key] = {}
                else:
                    report_data[key] = []

        # immediate_actions 검증
        for i, action in enumerate(report_data.get("immediate_actions", [])):
            required_fields = ["priority", "action", "why", "owner"]
            for field in required_fields:
                if field not in action:
                    self.logger.warning(
                        f"immediate_actions[{i}] missing '{field}', adding default"
                    )
                    action[field] = "unknown"

        return report_data

    def export_to_markdown(self, report: Dict[str, Any]) -> str:
        """
        리포트를 마크다운 형식으로 변환

        Args:
            report: execute() 결과의 report

        Returns:
            마크다운 형식의 리포트 문자열
        """
        md_lines = []

        # 헤더
        metadata = report.get("metadata", {})
        md_lines.append("# 📊 리뷰 분석 리포트")
        md_lines.append("")
        md_lines.append(f"**제품 ID**: {metadata.get('product_id', 'N/A')}")
        md_lines.append(f"**분석 일시**: {metadata.get('generated_at', 'N/A')}")
        md_lines.append(f"**총 리뷰 수**: {metadata.get('total_reviews', 0):,}")
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")

        # Executive Summary
        md_lines.append("## 📌 Executive Summary")
        md_lines.append("")
        md_lines.append(report.get("executive_summary", "N/A"))
        md_lines.append("")

        # Key Findings
        md_lines.append("## 🔍 Key Findings")
        md_lines.append("")
        for finding in report.get("key_findings", []):
            md_lines.append(f"- {finding}")
        md_lines.append("")

        # Immediate Actions
        md_lines.append("## ⚡ Immediate Actions Required")
        md_lines.append("")
        for action in report.get("immediate_actions", []):
            priority = action.get("priority", "?")
            action_text = action.get("action", "N/A")
            why = action.get("why", "N/A")
            owner = action.get("owner", "N/A")

            md_lines.append(f"### Priority #{priority}")
            md_lines.append(f"**Action**: {action_text}")
            md_lines.append(f"**Why**: {why}")
            md_lines.append(f"**Owner**: {owner}")
            md_lines.append("")

        # Business Impact
        md_lines.append("## 💼 Business Impact")
        md_lines.append("")
        impact = report.get("business_impact", {})
        md_lines.append(f"**Current State**: {impact.get('current_state', 'N/A')}")
        md_lines.append(f"**Predicted Improvement**: {impact.get('predicted_improvement', 'N/A')}")
        md_lines.append(f"**Risk if Ignored**: {impact.get('risk_if_ignored', 'N/A')}")
        md_lines.append(f"**Estimated Timeline**: {impact.get('estimated_timeline', 'N/A')}")
        md_lines.append("")

        # Metrics to Track
        md_lines.append("## 📈 Metrics to Track")
        md_lines.append("")
        for metric in report.get("metrics_to_track", []):
            md_lines.append(f"- {metric}")
        md_lines.append("")

        return "\n".join(md_lines)

    def export_to_json(self, report: Dict[str, Any], file_path: str) -> None:
        """
        리포트를 JSON 파일로 저장

        Args:
            report: execute() 결과의 report
            file_path: 저장할 파일 경로
        """
        import json

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Report exported to JSON: {file_path}")

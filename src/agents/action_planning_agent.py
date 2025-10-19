"""
실행 계획 수립 에이전트 (Action Planning Agent)

Phase 2: 인사이트 기반 실행 가능한 개선 계획 생성

주요 기능:
- 추출된 pain point를 비즈니스 우선순위로 변환
- Quick Win / Medium-term / Long-term 액션 분류
- 담당 팀, 노력/영향도, 타임라인 제시
- LLM 활용 (gpt-oss:20b)
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from jinja2 import Template

from core.base_agent import BaseAgent
from core.exceptions import AgentExecutionError, LLMAPIError
from services.llm_service import LLMService
from utils.error_handler import retry_on_error, log_execution_time


class ActionPlanningAgent(BaseAgent):
    """
    실행 계획 수립 에이전트

    인사이트를 기반으로 우선순위가 지정된 실행 계획 생성
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
        template_path = Path(__file__).parent.parent / "prompts" / "action_planning.jinja2"
        if not template_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            self.prompt_template = Template(f.read())

        self.logger.info(f"ActionPlanningAgent initialized (v{self.VERSION})")

    @log_execution_time
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        실행 계획 생성

        Args:
            input_data: {
                "pain_points": List[Dict],  # InsightAgent 결과
                "stats": Dict,               # 기본 통계
                "product_id": str            # (선택) 제품 ID
            }

        Returns:
            {
                "action_plan": Dict,  # 전체 실행 계획
                "priorities": List[Dict],
                "quick_wins": List[Dict],
                "medium_term_actions": List[Dict],
                "long_term_actions": List[Dict],
                "expected_outcomes": Dict,
                "success": bool,
                "error": Optional[str]
            }
        """
        try:
            self.logger.info("Starting action planning...")

            # 입력 데이터 검증
            self._validate_input(input_data)

            pain_points = input_data["pain_points"]
            stats = input_data["stats"]

            # pain point가 없으면 조기 종료
            if len(pain_points) == 0:
                self.logger.warning("No pain points to plan for")
                return self._create_empty_plan()

            # 프롬프트 준비
            prompt = self._prepare_prompt(pain_points, stats)

            # LLM 호출 (JSON 모드)
            action_plan = self._call_llm(prompt)

            # 결과 검증 및 후처리
            plan = self._process_action_plan(action_plan)

            # 메트릭 기록
            self.metrics["priorities_count"] = len(plan.get("priorities", []))
            self.metrics["quick_wins_count"] = len(plan.get("quick_wins", []))
            self.metrics["medium_term_count"] = len(plan.get("medium_term_actions", []))
            self.metrics["long_term_count"] = len(plan.get("long_term_actions", []))
            self.metrics["total_actions"] = (
                self.metrics["quick_wins_count"] +
                self.metrics["medium_term_count"] +
                self.metrics["long_term_count"]
            )

            self.logger.info(
                f"Action planning completed: "
                f"{self.metrics['total_actions']} total actions "
                f"({self.metrics['quick_wins_count']} quick wins)"
            )

            return {
                "action_plan": plan,
                "priorities": plan.get("priorities", []),
                "quick_wins": plan.get("quick_wins", []),
                "medium_term_actions": plan.get("medium_term_actions", []),
                "long_term_actions": plan.get("long_term_actions", []),
                "expected_outcomes": plan.get("expected_outcomes", {}),
                "success": True,
                "error": None
            }

        except Exception as e:
            self.logger.error(f"Action planning failed: {str(e)}", exc_info=True)
            raise AgentExecutionError(f"ActionPlanningAgent failed: {str(e)}")

    def _validate_input(self, input_data: Dict[str, Any]) -> None:
        """입력 데이터 검증"""
        required_keys = ["pain_points", "stats"]
        for key in required_keys:
            if key not in input_data:
                raise ValueError(f"Missing required key: {key}")

        if not isinstance(input_data["pain_points"], list):
            raise TypeError("'pain_points' must be a list")

    def _prepare_prompt(
        self,
        pain_points: List[Dict],
        stats: Dict
    ) -> str:
        """
        Jinja2 템플릿을 사용하여 프롬프트 생성
        """
        # 부정 리뷰 비율 계산
        total_reviews = stats.get("total_reviews", 0)
        negative_count = stats.get("negative_count", 0)
        negative_rate = (negative_count / total_reviews * 100) if total_reviews > 0 else 0

        # 템플릿 렌더링
        prompt = self.prompt_template.render(
            pain_points=pain_points,
            total_reviews=total_reviews,
            avg_rating=stats.get("avg_rating", 0),
            negative_rate=round(negative_rate, 1)
        )

        self.logger.debug(f"Prompt prepared: {len(prompt)} characters")
        return prompt

    @retry_on_error(max_retries=3, delay=2)
    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        LLM 호출 (retry 포함)

        gpt-oss:20b의 thinking 기능을 활용하여 전략적 계획 수립
        대규모 인사이트 처리를 위해 max_tokens 동적 조정
        """
        try:
            self.logger.info("Calling LLM for action planning...")

            # 대규모 분석을 위해 max_tokens 증가
            max_tokens = self.config.get("max_tokens", 3000)  # 2000 → 3000

            # JSON 모드로 호출
            plan = self.llm_service.generate_json(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=self.config.get("temperature", 0.7),
                retries=5  # 재시도 횟수 증가 (3 → 5)
            )

            if plan is None:
                raise LLMAPIError("LLM returned None response after all retries")

            self.logger.info("LLM call successful")
            return plan

        except Exception as e:
            self.logger.error(f"LLM call failed: {str(e)}")
            raise LLMAPIError(f"Failed to call LLM: {str(e)}")

    def _process_action_plan(self, plan_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM 응답 후처리 및 검증
        """
        # 필수 키 확인
        expected_keys = [
            "priorities",
            "quick_wins",
            "medium_term_actions",
            "long_term_actions",
            "expected_outcomes"
        ]

        for key in expected_keys:
            if key not in plan_data:
                self.logger.warning(f"Missing '{key}' in LLM response, adding empty")
                if key == "expected_outcomes":
                    plan_data[key] = {}
                else:
                    plan_data[key] = []

        # 각 액션 검증
        for action_type in ["quick_wins", "medium_term_actions", "long_term_actions"]:
            for i, action in enumerate(plan_data.get(action_type, [])):
                required_fields = ["action", "responsible_team", "effort", "impact"]
                for field in required_fields:
                    if field not in action:
                        self.logger.warning(
                            f"{action_type}[{i}] missing '{field}', adding default"
                        )
                        action[field] = "unknown"

        return plan_data

    def _create_empty_plan(self) -> Dict[str, Any]:
        """pain point가 없을 때 빈 계획 반환"""
        return {
            "action_plan": {},
            "priorities": [],
            "quick_wins": [],
            "medium_term_actions": [],
            "long_term_actions": [],
            "expected_outcomes": {},
            "success": True,
            "error": None,
            "message": "No pain points to plan for"
        }

    def get_high_impact_actions(
        self,
        action_plan: Dict[str, Any],
        min_impact: str = "medium"
    ) -> List[Dict[str, Any]]:
        """
        고영향 액션만 필터링

        Args:
            action_plan: execute() 결과의 action_plan
            min_impact: 최소 영향도 ("low", "medium", "high")

        Returns:
            필터링된 액션 리스트
        """
        impact_order = {"low": 1, "medium": 2, "high": 3}
        min_score = impact_order.get(min_impact, 2)

        high_impact = []

        for action_type in ["quick_wins", "medium_term_actions", "long_term_actions"]:
            for action in action_plan.get(action_type, []):
                impact_score = impact_order.get(action.get("impact", "low"), 1)
                if impact_score >= min_score:
                    high_impact.append(action)

        return high_impact

    def prioritize_by_roi(
        self,
        action_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        ROI (Impact / Effort) 기준으로 액션 우선순위 정렬

        Args:
            action_plan: execute() 결과의 action_plan

        Returns:
            ROI 순으로 정렬된 액션 리스트
        """
        effort_score = {"low": 1, "medium": 2, "high": 3}
        impact_score = {"low": 1, "medium": 2, "high": 3}

        all_actions = []

        for action_type in ["quick_wins", "medium_term_actions", "long_term_actions"]:
            for action in action_plan.get(action_type, []):
                effort = effort_score.get(action.get("effort", "medium"), 2)
                impact = impact_score.get(action.get("impact", "medium"), 2)

                # ROI = Impact / Effort (높을수록 좋음)
                roi = impact / effort if effort > 0 else 0

                action_with_roi = action.copy()
                action_with_roi["roi"] = round(roi, 2)
                action_with_roi["action_type"] = action_type

                all_actions.append(action_with_roi)

        # ROI 내림차순 정렬
        sorted_actions = sorted(all_actions, key=lambda x: x["roi"], reverse=True)

        return sorted_actions

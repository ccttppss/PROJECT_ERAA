"""
모든 에이전트의 기본 추상 클래스
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from utils.logger import get_logger
from core.exceptions import AgentExecutionError


class BaseAgent(ABC):
    """모든 에이전트가 상속받아야 하는 기본 클래스"""

    VERSION = "1.0.0"

    def __init__(self, config: Dict[str, Any], logger=None):
        """
        Args:
            config: 에이전트 설정 딕셔너리
            logger: 로거 인스턴스 (선택)
        """
        self.config = config
        self.logger = logger or get_logger(
            self.__class__.__name__,
            level=config.get('log_level', 'INFO')
        )
        self.metrics: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, input_data: Any) -> Dict[str, Any]:
        """
        에이전트의 메인 실행 로직 (반드시 구현)

        Args:
            input_data: 입력 데이터

        Returns:
            실행 결과 딕셔너리

        Raises:
            AgentExecutionError: 실행 중 오류 발생 시
        """
        pass

    def validate_input(self, input_data: Any) -> bool:
        """
        입력 데이터 검증 (선택적 오버라이드)

        Args:
            input_data: 검증할 입력 데이터

        Returns:
            검증 성공 여부
        """
        return True

    def log_metrics(self, metric_name: str, value: Any):
        """
        메트릭 로깅

        Args:
            metric_name: 메트릭 이름
            value: 메트릭 값
        """
        self.metrics[metric_name] = value
        self.logger.info(f"Metric: {metric_name}", value=value)

    def get_version_info(self) -> Dict[str, str]:
        """
        에이전트 버전 정보 반환

        Returns:
            버전 정보 딕셔너리
        """
        return {
            "agent": self.__class__.__name__,
            "version": self.VERSION
        }

    def reset_metrics(self):
        """메트릭 초기화"""
        self.metrics = {}

    def get_metrics(self) -> Dict[str, Any]:
        """
        수집된 메트릭 반환

        Returns:
            메트릭 딕셔너리
        """
        return self.metrics.copy()

    def _log_execution_start(self):
        """실행 시작 로그"""
        self.logger.info(f"{self.__class__.__name__} execution started")

    def _log_execution_end(self, success: bool = True):
        """실행 종료 로그"""
        status = "completed" if success else "failed"
        self.logger.info(f"{self.__class__.__name__} execution {status}")

"""
로깅 설정 및 구조화된 로거
"""
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class StructuredLogger:
    """구조화된 로깅을 제공하는 로거 클래스"""

    def __init__(self, name: str, level: str = "INFO", log_file: Optional[str] = None):
        """
        Args:
            name: 로거 이름
            level: 로깅 레벨 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: 로그 파일 경로 (선택)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # 중복 핸들러 방지
        if not self.logger.handlers:
            # 콘솔 핸들러
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

            # 파일 핸들러 (선택)
            if log_file:
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
                self.logger.addHandler(file_handler)

    def debug(self, message: str, **kwargs):
        """디버그 로그"""
        self.logger.debug(self._format_message(message, kwargs))

    def info(self, message: str, **kwargs):
        """정보 로그"""
        self.logger.info(self._format_message(message, kwargs))

    def warning(self, message: str, **kwargs):
        """경고 로그"""
        self.logger.warning(self._format_message(message, kwargs))

    def error(self, message: str, **kwargs):
        """에러 로그"""
        self.logger.error(self._format_message(message, kwargs))

    def critical(self, message: str, **kwargs):
        """심각한 오류 로그"""
        self.logger.critical(self._format_message(message, kwargs))

    def log_agent_execution(
        self,
        agent_name: str,
        input_size: int,
        output_size: int,
        duration_ms: float
    ):
        """에이전트 실행 로그"""
        self.info(
            f"Agent executed: {agent_name}",
            input_size=input_size,
            output_size=output_size,
            duration_ms=duration_ms
        )

    def log_llm_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float
    ):
        """LLM API 호출 로그"""
        self.info(
            "LLM API called",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=cost_usd
        )

    def _format_message(self, message: str, extra: Dict[str, Any]) -> str:
        """메시지에 추가 정보 포함"""
        if extra:
            extra_str = " | " + " | ".join([f"{k}={v}" for k, v in extra.items()])
            return message + extra_str
        return message


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None
) -> StructuredLogger:
    """
    로거 인스턴스 생성 헬퍼 함수

    Args:
        name: 로거 이름
        level: 로깅 레벨
        log_file: 로그 파일 경로

    Returns:
        StructuredLogger 인스턴스
    """
    return StructuredLogger(name, level, log_file)

"""
커스텀 예외 클래스 정의
"""


class ReviewAnalysisException(Exception):
    """모든 커스텀 예외의 기본 클래스"""
    pass


class DataLoadError(ReviewAnalysisException):
    """데이터 로드 실패 시 발생"""
    pass


class DataValidationError(ReviewAnalysisException):
    """데이터 검증 실패 시 발생"""
    pass


class AgentExecutionError(ReviewAnalysisException):
    """에이전트 실행 중 오류 발생 시"""
    pass


class LLMAPIError(ReviewAnalysisException):
    """LLM API 호출 오류"""
    pass


class ConfigurationError(ReviewAnalysisException):
    """설정 파일 오류"""
    pass


class InsufficientDataError(ReviewAnalysisException):
    """분석하기에 데이터가 부족한 경우"""
    pass

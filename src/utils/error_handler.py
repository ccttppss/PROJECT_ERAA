"""
에러 처리 유틸리티 및 데코레이터
"""
import time
import functools
from typing import Callable, Any, Type, Tuple, Optional
from utils.logger import get_logger

logger = get_logger(__name__)


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    에러 발생 시 재시도하는 데코레이터

    Args:
        max_retries: 최대 재시도 횟수
        delay: 초기 대기 시간 (초)
        backoff: 대기 시간 증가 배수 (exponential backoff)
        exceptions: 재시도할 예외 타입들

    Example:
        @retry_on_error(max_retries=3, delay=1.0, backoff=2.0)
        def unstable_function():
            # 불안정한 작업
            pass
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries - 1:
                        logger.error(
                            f"Function {func.__name__} failed after {max_retries} attempts",
                            error=str(e)
                        )
                        raise

                    logger.warning(
                        f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries}). "
                        f"Retrying in {current_delay}s...",
                        error=str(e)
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff

            return None  # 이 라인은 실행되지 않지만 타입 체커를 위해 유지

        return wrapper
    return decorator


def log_execution_time(func: Callable) -> Callable:
    """
    함수 실행 시간을 로깅하는 데코레이터

    Example:
        @log_execution_time
        def slow_function():
            # 시간이 오래 걸리는 작업
            pass
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        result = func(*args, **kwargs)
        duration_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Function {func.__name__} completed",
            duration_ms=round(duration_ms, 2)
        )

        return result

    return wrapper


def safe_execute(
    func: Callable,
    default_return: Any = None,
    log_error: bool = True
) -> Any:
    """
    안전하게 함수를 실행하고 에러 시 기본값 반환

    Args:
        func: 실행할 함수
        default_return: 에러 시 반환할 기본값
        log_error: 에러 로깅 여부

    Returns:
        함수 실행 결과 또는 기본값
    """
    try:
        return func()
    except Exception as e:
        if log_error:
            logger.error(
                f"Error in safe_execute for {func.__name__}",
                error=str(e)
            )
        return default_return


class ErrorHandler:
    """에러 처리를 위한 컨텍스트 매니저"""

    def __init__(
        self,
        error_message: str,
        raise_error: bool = True,
        default_return: Any = None
    ):
        """
        Args:
            error_message: 에러 메시지
            raise_error: 에러를 다시 발생시킬지 여부
            default_return: raise_error=False일 때 반환할 기본값
        """
        self.error_message = error_message
        self.raise_error = raise_error
        self.default_return = default_return
        self.exception: Optional[Exception] = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.exception = exc_val
            logger.error(
                self.error_message,
                error_type=exc_type.__name__,
                error=str(exc_val)
            )

            if not self.raise_error:
                return True  # 예외 억제

        return False  # 예외 전파


class RateLimiter:
    """
    Rate Limiting 구현 (Token Bucket 알고리즘, Phase 3)

    일정 시간 동안 최대 N개 요청만 허용하여 API 과부하 방지
    """

    def __init__(self, max_calls: int = 10, time_window: float = 60.0):
        """
        Args:
            max_calls: 시간 윈도우당 최대 호출 횟수
            time_window: 시간 윈도우 (초)
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

        logger.info(
            f"RateLimiter initialized | max_calls={max_calls} | window={time_window}s"
        )

    def acquire(self):
        """
        요청 토큰 획득 (필요시 대기)

        Rate limit을 초과하면 시간 윈도우가 지날 때까지 대기합니다.
        """
        current_time = time.time()

        # 시간 윈도우 밖의 호출 기록 제거
        self.calls = [call_time for call_time in self.calls
                      if current_time - call_time < self.time_window]

        # Rate limit 초과 시 대기
        if len(self.calls) >= self.max_calls:
            # 가장 오래된 호출이 윈도우를 벗어날 때까지 대기
            oldest_call = self.calls[0]
            wait_time = self.time_window - (current_time - oldest_call)

            if wait_time > 0:
                logger.warning(
                    f"Rate limit reached. Waiting {wait_time:.2f}s... "
                    f"(calls={len(self.calls)}/{self.max_calls})"
                )
                time.sleep(wait_time)

                # 대기 후 다시 정리
                current_time = time.time()
                self.calls = [call_time for call_time in self.calls
                             if current_time - call_time < self.time_window]

        # 현재 호출 기록
        self.calls.append(current_time)

        logger.debug(
            f"Rate limiter token acquired | current_calls={len(self.calls)}/{self.max_calls}"
        )

    def reset(self):
        """Rate limiter 초기화"""
        self.calls = []
        logger.info("Rate limiter reset")


def rate_limited(max_calls: int = 10, time_window: float = 60.0):
    """
    Rate limiting 데코레이터 (Phase 3)

    Args:
        max_calls: 시간 윈도우당 최대 호출 횟수
        time_window: 시간 윈도우 (초)

    Example:
        @rate_limited(max_calls=5, time_window=60.0)
        def api_call():
            # API 호출 (분당 최대 5회)
            pass
    """
    limiter = RateLimiter(max_calls, time_window)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            limiter.acquire()
            return func(*args, **kwargs)

        return wrapper

    return decorator

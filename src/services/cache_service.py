"""
캐시 서비스 (Phase 3)

기능:
- LLM 응답 캐싱 (동일 프롬프트 재사용)
- 파일 기반 영구 저장
- TTL (Time To Live) 지원
- 캐시 히트/미스 통계
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Any, Optional, Dict
import logging


class CacheService:
    """
    LLM 응답 캐시 서비스 (Phase 3)

    프롬프트 해시를 키로 사용하여 LLM 응답을 캐싱합니다.
    동일한 프롬프트에 대해 재사용하여 속도 향상 및 비용 절감.
    """

    VERSION = "3.0.0"

    def __init__(
        self,
        cache_dir: str = "output/cache",
        ttl_seconds: int = 86400,  # 24시간
        logger: Optional[logging.Logger] = None
    ):
        """
        Args:
            cache_dir: 캐시 파일 저장 디렉토리
            ttl_seconds: Time To Live (초). 기본 24시간
            logger: 로거 인스턴스
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttl_seconds = ttl_seconds
        self.logger = logger or self._get_default_logger()

        self.cache_file = self.cache_dir / "llm_cache.json"

        # 캐시 데이터 로드
        self.cache_data = self._load_cache()

        # 통계
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }

        self.logger.info(
            f"CacheService initialized (v{self.VERSION}) | "
            f"cache_dir={self.cache_dir} | ttl={ttl_seconds}s | "
            f"cache_entries={len(self.cache_data)}"
        )

    def _get_default_logger(self):
        """기본 로거 생성"""
        logger = logging.getLogger("CacheService")
        logger.setLevel(logging.INFO)
        return logger

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """캐시 파일 로드"""
        if not self.cache_file.exists():
            return {}

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            return {}

    def _save_cache(self):
        """캐시 파일 저장"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save cache: {e}")

    def _generate_key(self, prompt: str, model: str = "default") -> str:
        """
        프롬프트에서 캐시 키 생성 (해시)

        Args:
            prompt: 프롬프트 텍스트
            model: 모델 이름 (키에 포함)

        Returns:
            SHA-256 해시 문자열
        """
        content = f"{model}:{prompt}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _is_expired(self, timestamp: float) -> bool:
        """
        캐시 항목 만료 여부 확인

        Args:
            timestamp: 저장 시각 (Unix timestamp)

        Returns:
            만료 여부
        """
        if self.ttl_seconds <= 0:
            return False  # TTL 비활성화

        age_seconds = time.time() - timestamp
        return age_seconds > self.ttl_seconds

    def get(self, prompt: str, model: str = "default") -> Optional[str]:
        """
        캐시에서 응답 조회

        Args:
            prompt: 프롬프트 텍스트
            model: 모델 이름

        Returns:
            캐시된 응답 (없거나 만료시 None)
        """
        self.stats["total_requests"] += 1

        key = self._generate_key(prompt, model)

        if key not in self.cache_data:
            self.stats["misses"] += 1
            self.logger.debug(f"Cache MISS | key={key[:16]}...")
            return None

        entry = self.cache_data[key]

        # 만료 체크
        if self._is_expired(entry["timestamp"]):
            self.stats["misses"] += 1
            self.logger.debug(f"Cache EXPIRED | key={key[:16]}...")
            del self.cache_data[key]
            self._save_cache()
            return None

        # 캐시 히트!
        self.stats["hits"] += 1
        hit_rate = self.stats["hits"] / self.stats["total_requests"] * 100

        self.logger.info(
            f"Cache HIT | key={key[:16]}... | "
            f"hit_rate={hit_rate:.1f}% | age={(time.time() - entry['timestamp']) / 60:.1f}min"
        )

        return entry["response"]

    def set(self, prompt: str, response: str, model: str = "default"):
        """
        응답을 캐시에 저장

        Args:
            prompt: 프롬프트 텍스트
            response: LLM 응답
            model: 모델 이름
        """
        key = self._generate_key(prompt, model)

        self.cache_data[key] = {
            "response": response,
            "timestamp": time.time(),
            "model": model,
            "prompt_preview": prompt[:100] + "..." if len(prompt) > 100 else prompt
        }

        self._save_cache()

        self.logger.debug(f"Cache SET | key={key[:16]}... | size={len(response)} chars")

    def clear(self, expired_only: bool = False):
        """
        캐시 초기화

        Args:
            expired_only: True이면 만료된 항목만 삭제, False이면 전체 삭제
        """
        if not expired_only:
            self.cache_data = {}
            self._save_cache()
            self.logger.info("Cache cleared (all entries)")
            return

        # 만료된 항목만 삭제
        initial_count = len(self.cache_data)
        expired_keys = [
            key for key, entry in self.cache_data.items()
            if self._is_expired(entry["timestamp"])
        ]

        for key in expired_keys:
            del self.cache_data[key]

        self._save_cache()

        self.logger.info(
            f"Cache cleared (expired only) | "
            f"removed={len(expired_keys)} | remaining={len(self.cache_data)}"
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        캐시 통계 반환

        Returns:
            통계 딕셔너리
        """
        total = self.stats["total_requests"]
        hit_rate = (self.stats["hits"] / total * 100) if total > 0 else 0

        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "total_requests": total,
            "hit_rate_percent": round(hit_rate, 2),
            "cache_entries": len(self.cache_data)
        }

    def print_stats(self):
        """캐시 통계 출력 (로그)"""
        stats = self.get_stats()

        self.logger.info("=" * 60)
        self.logger.info("📊 Cache Statistics")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Requests: {stats['total_requests']}")
        self.logger.info(f"Cache Hits: {stats['hits']}")
        self.logger.info(f"Cache Misses: {stats['misses']}")
        self.logger.info(f"Hit Rate: {stats['hit_rate_percent']}%")
        self.logger.info(f"Cache Entries: {stats['cache_entries']}")
        self.logger.info("=" * 60)

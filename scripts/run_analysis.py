"""
리뷰 분석 시스템 실행 스크립트 (Phase 1)
"""
import argparse
import sys
from pathlib import Path

# src 폴더를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from core.orchestrator import ReviewAnalysisOrchestrator
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(
        description='전자상거래 리뷰 분석 시스템 (Phase 1)'
    )

    parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='리뷰 데이터 파일 경로 (JSON)'
    )

    parser.add_argument(
        '--product-id',
        type=str,
        default=None,
        help='분석할 제품 ID (ASIN). 미지정 시 전체 분석'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=1000,
        help='분석할 최대 리뷰 수 (기본값: 1000)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='./reports/results.json',
        help='결과 저장 경로 (기본값: ./reports/results.json)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='로깅 레벨 (기본값: INFO)'
    )

    args = parser.parse_args()

    # 설정 생성
    config = {
        'log_level': args.log_level
    }

    try:
        # 오케스트레이터 초기화
        orchestrator = ReviewAnalysisOrchestrator(config)

        # 분석 실행
        results = orchestrator.run_analysis(
            data_path=args.data_path,
            product_id=args.product_id,
            limit=args.limit
        )

        # 결과 저장
        orchestrator.save_results(args.output)

        # 요약 출력
        print("\n" + "=" * 60)
        print(orchestrator.get_summary())
        print("=" * 60)

        logger.info(f"\n✅ 분석 완료! 결과: {args.output}")

        return 0

    except FileNotFoundError as e:
        logger.error(f"파일을 찾을 수 없습니다: {e}")
        return 1

    except Exception as e:
        logger.error(f"분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

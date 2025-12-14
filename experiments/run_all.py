"""
전체 실험 실행 스크립트

4가지 실험을 순차적으로 실행하고 결과를 종합합니다.
"""
import sys
from pathlib import Path

# src 폴더를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from experiments.ablation_study import run_ablation_study
from experiments.processing_time import run_processing_time_experiment
from experiments.sarcasm_detection import run_sarcasm_experiment
from experiments.mixed_sentiment import run_mixed_sentiment_experiment


def run_all_experiments():
    """모든 실험 실행"""
    print("\n" + "=" * 80)
    print("🔬 KCI 학술지 투고용 실험 전체 실행")
    print("=" * 80)
    
    results = {}
    
    # 실험 1: Ablation Study
    print("\n\n")
    try:
        results['ablation'] = run_ablation_study()
    except Exception as e:
        print(f"❌ 실험 1 실패: {e}")
        results['ablation'] = {'error': str(e)}
    
    # 실험 2: 처리 시간
    print("\n\n")
    try:
        results['processing_time'] = run_processing_time_experiment()
    except Exception as e:
        print(f"❌ 실험 2 실패: {e}")
        results['processing_time'] = {'error': str(e)}
    
    # 실험 3: 반어법
    print("\n\n")
    try:
        results['sarcasm'] = run_sarcasm_experiment()
    except Exception as e:
        print(f"❌ 실험 3 실패: {e}")
        results['sarcasm'] = {'error': str(e)}
    
    # 실험 4: 혼합 감정
    print("\n\n")
    try:
        results['mixed_sentiment'] = run_mixed_sentiment_experiment()
    except Exception as e:
        print(f"❌ 실험 4 실패: {e}")
        results['mixed_sentiment'] = {'error': str(e)}
    
    # 최종 요약
    print("\n\n" + "=" * 80)
    print("📊 전체 실험 결과 요약")
    print("=" * 80)
    
    print("\n실험 1: Ablation Study")
    if 'error' not in results.get('ablation', {}):
        for name, r in results['ablation'].items():
            print(f"   - {name}: {r.get('accuracy', 0):.2%}")
    else:
        print(f"   - 오류: {results['ablation'].get('error')}")
    
    print("\n실험 2: 처리 시간")
    if 'error' not in results.get('processing_time', {}):
        print("   - 결과 저장 완료")
    else:
        print(f"   - 오류: {results['processing_time'].get('error')}")
    
    print("\n실험 3: 반어법 탐지")
    if 'error' not in results.get('sarcasm', {}):
        if 'llm_based' in results['sarcasm']:
            sarcasm_acc = results['sarcasm']['llm_based'].get('sarcasm_accuracy', 0)
            print(f"   - LLM 반어법 정확도: {sarcasm_acc:.2%}")
    else:
        print(f"   - 오류: {results['sarcasm'].get('error')}")
    
    print("\n실험 4: 혼합 감정")
    if 'error' not in results.get('mixed_sentiment', {}):
        if 'llm_based' in results['mixed_sentiment']:
            mixed_acc = results['mixed_sentiment']['llm_based'].get('accuracy', 0)
            print(f"   - LLM Aspect-level 정확도: {mixed_acc:.2%}")
    else:
        print(f"   - 오류: {results['mixed_sentiment'].get('error')}")
    
    print("\n" + "=" * 80)
    print("✅ 모든 실험 완료!")
    print(f"📁 결과 위치: {project_root / 'experiments' / 'results'}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    run_all_experiments()

"""
실험 3: 반어법 (Sarcasm) 탐지 - SARC 데이터셋 사용

목적: LLM이 반어법을 얼마나 잘 감지하는지 측정
데이터: SARC (Self-Annotated Reddit Corpus) - LREC 2018

SARC 특징: 
- Reddit 댓글 기반
- 작성자가 직접 /s 태그로 sarcasm 표시
- 1M+ 댓글 (balanced: 50/50)
"""
import sys
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Any

# src 폴더를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from services.llm_service import create_llm_service
from utils.logger import get_logger
import yaml
from jinja2 import Template

logger = get_logger(__name__)


def load_llm_service():
    """LLM 서비스 로드"""
    config_path = project_root / 'src' / 'config' / 'llm_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        llm_config = yaml.safe_load(f)
    return create_llm_service(llm_config)


def load_sarc_dataset(max_samples: int = 500) -> List[Dict]:
    """
    SARC 데이터셋 로드
    
    Args:
        max_samples: 최대 샘플 수 (balanced: sarcastic/non-sarcastic 반반)
    
    Returns:
        List of {comment, label, parent_comment}
    """
    # train-balanced-sarcasm.csv 사용 (더 깨끗한 포맷)
    sarc_path = project_root / 'datasets' / 'sarc' / 'train-balanced-sarcasm.csv'
    
    if not sarc_path.exists():
        # test-balanced.csv 시도
        sarc_path = project_root / 'datasets' / 'sarc' / 'test-balanced.csv'
    
    if not sarc_path.exists():
        raise FileNotFoundError(f"SARC 데이터셋 없음: {sarc_path}")
    
    samples = {'sarcastic': [], 'non_sarcastic': []}
    
    with open(sarc_path, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            comment = row.get('comment', '')
            label = row.get('label', '')
            parent = row.get('parent_comment', '')
            
            if not comment or len(comment) < 10:
                continue
            
            if len(comment) > 500:  # 너무 긴 댓글 제외
                continue
            
            sample = {
                'comment': comment,
                'label': int(label) if label.isdigit() else 0,
                'parent_comment': parent
            }
            
            if sample['label'] == 1:
                samples['sarcastic'].append(sample)
            else:
                samples['non_sarcastic'].append(sample)
            
            # 충분히 모았으면 중단
            if len(samples['sarcastic']) >= max_samples and len(samples['non_sarcastic']) >= max_samples:
                break
    
    # Balanced 샘플링
    n_each = max_samples // 2
    selected = []
    
    if len(samples['sarcastic']) >= n_each:
        selected.extend(random.sample(samples['sarcastic'], n_each))
    else:
        selected.extend(samples['sarcastic'])
    
    if len(samples['non_sarcastic']) >= n_each:
        selected.extend(random.sample(samples['non_sarcastic'], n_each))
    else:
        selected.extend(samples['non_sarcastic'])
    
    random.shuffle(selected)
    return selected


def evaluate_surface_sentiment(samples: List[Dict]) -> Dict:
    """
    표면적 감정 기반 평가 (반어법 구분 불가)
    
    Sarcastic 댓글은 표면적으로 긍정적이지만 실제는 부정적
    → 표면 감정만 보면 sarcasm을 놓침
    """
    # 간단한 규칙 기반 감정 분석
    positive_words = {'great', 'love', 'amazing', 'awesome', 'good', 'best', 'wonderful', 'excellent', 'perfect', 'nice'}
    negative_words = {'bad', 'terrible', 'awful', 'worst', 'hate', 'horrible', 'disgusting', 'poor', 'fail', 'sucks'}
    
    correct = 0
    total = len(samples)
    
    for sample in samples:
        text = sample['comment'].lower()
        
        # 표면적 감정 판단
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        
        if pos_count > neg_count:
            predicted_sarcasm = 0  # 긍정 표현 → sarcasm이 아니라고 예측
        elif neg_count > pos_count:
            predicted_sarcasm = 0  # 부정 표현 → 직접적 부정이므로 sarcasm 아님
        else:
            predicted_sarcasm = 0  # 중립 → sarcasm이 아니라고 예측
        
        if predicted_sarcasm == sample['label']:
            correct += 1
    
    return {
        'method': 'surface_sentiment',
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total
    }


def evaluate_llm_based(samples: List[Dict], llm_service, max_samples: int = 500) -> Dict:
    """LLM 기반 반어법 탐지"""
    if not llm_service:
        return {'method': 'llm_based', 'accuracy': 0, 'correct': 0, 'total': 0}
    
    eval_samples = samples[:max_samples]
    
    sarcasm_prompt = Template("""
다음 Reddit 댓글이 문맥(부모 댓글)을 고려했을 때 반어법(sarcasm)인지 판단하세요.

반어법은 표면적으로는 긍정적이거나 칭찬하는 것처럼 보이지만, 
실제 의도는 비꼼/조롱/비판인 표현입니다. 부모 댓글에 대한 반응임을 고려하세요.

부모 댓글 (Context): "{{ parent }}"
대상 댓글 (Reply): "{{ comment }}"

반드시 JSON 형식으로만 답하세요:
{"is_sarcasm": true 또는 false, "confidence": 0.0-1.0, "reason": "간단한 이유"}
""")
    
    correct = 0
    total = 0
    sarcasm_correct = 0
    sarcasm_total = 0
    details = []
    
    for idx, sample in enumerate(eval_samples):
        try:
            prompt = sarcasm_prompt.render(
                comment=sample['comment'][:300],
                parent=sample['parent_comment'][:300]
            )
            response = llm_service.generate_json(prompt, max_tokens=10000, temperature=0.3)
            
            # 요청 간 딜레이 (Ollama 과부하 방지)
            time.sleep(2.0)  # 1초 → 2초로 증가
            
            if response and 'is_sarcasm' in response:
                predicted = 1 if response['is_sarcasm'] else 0
            else:
                predicted = 0  # 폴백
        except Exception as e:
            logger.warning(f"LLM 호출 실패: {e}")
            predicted = 0
            # 오류 발생 시 추가 대기
            time.sleep(5.0)
        
        actual = sample['label']
        is_correct = predicted == actual
        
        if is_correct:
            correct += 1
        
        if actual == 1:  # 실제 sarcasm인 경우
            sarcasm_total += 1
            if is_correct:
                sarcasm_correct += 1
        
        total += 1
        
        details.append({
            'comment': sample['comment'][:60] + '...',
            'actual': actual,
            'predicted': predicted,
            'correct': is_correct
        })
        
        # 진행률 출력
        if (idx + 1) % 50 == 0:
            print(f"   진행: {idx + 1}/{len(eval_samples)}")
    
    return {
        'method': 'llm_based',
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
        'sarcasm_accuracy': sarcasm_correct / sarcasm_total if sarcasm_total > 0 else 0,
        'sarcasm_correct': sarcasm_correct,
        'sarcasm_total': sarcasm_total,
        'details': details
    }


def run_sarcasm_experiment(max_samples: int = 500):
    """SARC 반어법 탐지 실험 실행"""
    print("=" * 70)
    print("😏 실험 3: 반어법 탐지 (SARC Dataset)")
    print("=" * 70)
    print("📚 데이터: SARC (Self-Annotated Reddit Corpus)")
    print("📖 특징: Reddit 사용자가 /s 태그로 직접 라벨링")
    
    # 데이터셋 로드
    print(f"\n📂 SARC 데이터셋 로드 ({max_samples}개 샘플)...")
    try:
        samples = load_sarc_dataset(max_samples)
        sarcasm_count = sum(1 for s in samples if s['label'] == 1)
        
        print(f"   ✅ 로드 완료: {len(samples)}개")
        print(f"   ✅ Sarcastic: {sarcasm_count}")
        print(f"   ✅ Non-sarcastic: {len(samples) - sarcasm_count}")
    except FileNotFoundError as e:
        print(f"   ❌ {e}")
        return None
    
    # LLM 서비스 로드
    print("\n📡 LLM 서비스 로드...")
    try:
        llm_service = load_llm_service()
        print("   ✅ LLM service ready")
    except Exception as e:
        print(f"   ❌ LLM 로드 실패: {e}")
        llm_service = None
    
    results = {}
    
    # ===== 방법 A: 표면 감정 기반 =====
    print("\n" + "-" * 50)
    print("📌 방법 A: 표면 감정 기반 (반어법 구분 불가)")
    
    result_surface = evaluate_surface_sentiment(samples)
    results['surface'] = result_surface
    
    print(f"   Accuracy: {result_surface['accuracy']:.2%}")
    print(f"   정답: {result_surface['correct']}/{result_surface['total']}")
    
    # ===== 방법 B: LLM 기반 =====
    print("\n" + "-" * 50)
    print(f"📌 방법 B: LLM 기반 ({max_samples}개 샘플)")
    
    if llm_service:
        result_llm = evaluate_llm_based(samples, llm_service, max_samples)
        results['llm_based'] = result_llm
        
        print(f"\n   전체 Accuracy: {result_llm['accuracy']:.2%}")
        print(f"   Sarcasm Recall: {result_llm['sarcasm_accuracy']:.2%}")
        print(f"   정답: {result_llm['correct']}/{result_llm['total']}")
        
        # 샘플 결과 출력
        if 'details' in result_llm:
            print("\n   📋 샘플:")
            for detail in result_llm['details'][:3]:
                status = "✅" if detail['correct'] else "❌"
                label = "SARC" if detail['actual'] == 1 else "NORM"
                pred = "SARC" if detail['predicted'] == 1 else "NORM"
                print(f"      {status} [{label}→{pred}] \"{detail['comment']}\"")
    else:
        print("   ⚠️ LLM 서비스 없음")
    
    # ===== 결과 요약 =====
    print("\n" + "=" * 70)
    print("📊 SARC 반어법 탐지 결과")
    print("=" * 70)
    
    print(f"\n{'방법':<30} {'Accuracy':<15} {'Sarcasm Recall':<15}")
    print("-" * 60)
    print(f"{'A. 표면 감정 기반':<30} {result_surface['accuracy']:.2%}")
    
    if 'llm_based' in results:
        r = results['llm_based']
        print(f"{'B. LLM 기반':<30} {r['accuracy']:.2%}{'':<6} {r['sarcasm_accuracy']:.2%}")
        
        improvement = r['accuracy'] - result_surface['accuracy']
        print(f"\n📈 개선: +{improvement:.2%}")
    
    # 결과 저장
    output_path = project_root / 'experiments' / 'results' / 'sarc_sarcasm.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_results = {
        'dataset': 'SARC',
        'total_samples': len(samples),
        'surface': result_surface,
    }
    if 'llm_based' in results:
        save_results['llm_based'] = {
            k: v for k, v in results['llm_based'].items() if k != 'details'
        }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {output_path}")
    
    return results


if __name__ == "__main__":
    run_sarcasm_experiment(max_samples=1000)  # 1000개 샘플로 통계적 유의성 확보

"""
실험 1/4: ABSA 벤치마크 및 Ablation Study - SemEval 2014 데이터셋 사용

목적: 
1. SemEval 2014 표준 벤치마크에서 LLM vs 별점 기반 비교
2. 각 컴포넌트 기여도 분석 (Ablation)

데이터: SemEval 2014 Task 4 - Aspect Based Sentiment Analysis
- Laptop: 654 aspects (test)
- Restaurant: 1,134 aspects (test)
"""
import sys
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random
import time

# src 폴더를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from agents.sentiment_agent import SentimentAnalysisAgent
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


def load_semeval_dataset(domain: str = 'laptop', split: str = 'test') -> List[Dict]:
    """
    SemEval 2014 데이터셋 로드
    
    Args:
        domain: 'laptop' or 'restaurant'
        split: 'train' or 'test'
    
    Returns:
        List of {text, aspects: [{term, polarity, from, to}]}
    """
    filename = f"{domain}_{split}.xml"
    semeval_path = project_root / 'datasets' / 'semeval2014' / filename
    
    if not semeval_path.exists():
        raise FileNotFoundError(f"SemEval 데이터셋 없음: {semeval_path}")
    
    tree = ET.parse(semeval_path)
    root = tree.getroot()
    
    samples = []
    for sentence in root.findall('.//sentence'):
        text_elem = sentence.find('text')
        if text_elem is None:
            continue
        
        text = text_elem.text
        aspects = []
        
        for aspect_term in sentence.findall('.//aspectTerm'):
            term = aspect_term.get('term')
            polarity = aspect_term.get('polarity')
            from_idx = aspect_term.get('from')
            to_idx = aspect_term.get('to')
            
            # conflict와 unknown은 제외
            if polarity in ['positive', 'negative', 'neutral']:
                aspects.append({
                    'term': term,
                    'polarity': polarity,
                    'from': int(from_idx) if from_idx else 0,
                    'to': int(to_idx) if to_idx else 0
                })
        
        if aspects:
            samples.append({
                'text': text,
                'aspects': aspects
            })
    
    return samples


def evaluate_rating_based(samples: List[Dict]) -> Dict:
    """
    별점 기반 평가 (모든 aspect에 동일 감정)
    
    가정: 별점 없이 전체 리뷰 기준으로 majority 감정 예측
    """
    total_aspects = 0
    correct_aspects = 0
    
    # Baseline: 통계적으로 가장 많은 'positive'로 모두 예측 (Majority Class)
    majority = 'positive'
    
    for sample in samples:
        for aspect in sample['aspects']:
            total_aspects += 1
            if majority == aspect['polarity']:
                correct_aspects += 1
    
    return {
        'method': 'rating_based',
        'accuracy': correct_aspects / total_aspects if total_aspects > 0 else 0,
        'correct': correct_aspects,
        'total': total_aspects
    }


def evaluate_llm_based(samples: List[Dict], agent: SentimentAnalysisAgent,
                       max_samples: int = 100) -> Dict:
    """LLM 기반 aspect sentiment 분류"""
    if not agent.llm_service:
        return {'method': 'llm_based', 'accuracy': 0, 'correct': 0, 'total': 0}
    
    eval_samples = samples[:max_samples] if len(samples) > max_samples else samples
    
    total_aspects = 0
    correct_aspects = 0
    details = []
    
    for idx, sample in enumerate(eval_samples):
        text = sample['text']
        true_aspects = sample['aspects']
        
        # 실제 aspect term 리스트 추출
        aspect_terms = [a['term'] for a in true_aspects]
        
        try:
            sentiment_result = agent._llm_aspect_sentiment(text, aspect_terms)
            time.sleep(2.0)  # Ollama 과부하 방지
        except Exception as e:
            logger.warning(f"LLM 호출 실패: {e}")
            sentiment_result = {}
            time.sleep(5.0)
        
        for aspect in true_aspects:
            term = aspect['term'].lower()
            true_polarity = aspect['polarity']
            total_aspects += 1
            
            predicted = sentiment_result.get(term, 'not_found')
            if predicted in ['not_mentioned', 'unknown']:
                predicted = 'not_found'
            
            if predicted == true_polarity:
                correct_aspects += 1
        
        if (idx + 1) % 20 == 0:
            print(f"   진행: {idx + 1}/{len(eval_samples)}")
    
    return {
        'method': 'llm_based',
        'accuracy': correct_aspects / total_aspects if total_aspects > 0 else 0,
        'correct': correct_aspects,
        'total': total_aspects
    }


def evaluate_without_keyword(samples: List[Dict], llm_service,
                             max_samples: int = 100) -> Dict:
    """키워드 추출 없이 전체 리뷰 → LLM"""
    if not llm_service:
        return {'method': 'without_keyword', 'accuracy': 0, 'correct': 0, 'total': 0}
    
    eval_samples = samples[:max_samples]
    
    simple_template = Template("""
다음 리뷰에서 주어진 aspect term의 감정을 분석하세요.

리뷰: "{{ text }}"
Aspect: "{{ aspect }}"

positive, negative, neutral 중 하나로만 답하세요.
반드시 JSON 형식: {"sentiment": "positive|negative|neutral"}
""")
    
    total = 0
    correct = 0
    
    for idx, sample in enumerate(eval_samples):
        for aspect in sample['aspects']:
            try:
                prompt = simple_template.render(
                    text=sample['text'][:300],
                    aspect=aspect['term']
                )
                response = llm_service.generate_json(prompt, max_tokens=10000, temperature=0.3)
                time.sleep(1.0) # aspect 단위 호출이므로 짧게
                
                if response and 'sentiment' in response:
                    if response['sentiment'] == aspect['polarity']:
                        correct += 1
            except:
                pass
            total += 1
        
        if (idx + 1) % 20 == 0:
            print(f"   진행: {idx + 1}/{len(eval_samples)}")
    
    return {
        'method': 'without_keyword',
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total
    }


def run_semeval_experiment(domain: str = 'laptop', max_samples: int = 100):
    """SemEval 2014 ABSA 실험 실행"""
    print("=" * 70)
    print(f"🔬 실험 1: ABSA 벤치마크 (SemEval 2014 - {domain.capitalize()})")
    print("=" * 70)
    print("📚 데이터: SemEval 2014 Task 4")
    
    # 데이터셋 로드
    print(f"\n📂 SemEval 2014 {domain} 데이터셋 로드...")
    try:
        samples = load_semeval_dataset(domain, 'test')
        total_aspects = sum(len(s['aspects']) for s in samples)
        
        print(f"   ✅ 문장 수: {len(samples)}")
        print(f"   ✅ Aspect 수: {total_aspects}")
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
    
    config = {'log_level': 'WARNING'}
    agent = SentimentAnalysisAgent(config, llm_service)
    
    results = {}
    
    # ===== 방법 A: 별점/다수결 기반 =====
    print("\n" + "-" * 50)
    print("📌 조건 A: 별점 기반 (모든 aspect 동일 감정)")
    
    result_rating = evaluate_rating_based(samples)
    results['rating_based'] = result_rating
    
    print(f"   Accuracy: {result_rating['accuracy']:.2%}")
    print(f"   정답: {result_rating['correct']}/{result_rating['total']}")
    
    # ===== 방법 B: Full Model (LLM + 키워드) =====
    print("\n" + "-" * 50)
    print(f"📌 조건 B: Full Model - LLM + 키워드 ({max_samples}개 샘플)")
    
    if llm_service:
        result_llm = evaluate_llm_based(samples, agent, max_samples)
        results['full_model'] = result_llm
        
        print(f"\n   Accuracy: {result_llm['accuracy']:.2%}")
        print(f"   정답: {result_llm['correct']}/{result_llm['total']}")
    else:
        print("   ⚠️ LLM 서비스 없음")
    
    # ===== 방법 C: w/o 키워드 추출 =====
    print("\n" + "-" * 50)
    print(f"📌 조건 C: w/o 키워드 추출 ({max_samples}개 샘플)")
    
    if llm_service:
        result_no_kw = evaluate_without_keyword(samples, llm_service, max_samples)
        results['without_keyword'] = result_no_kw
        
        print(f"\n   Accuracy: {result_no_kw['accuracy']:.2%}")
        print(f"   정답: {result_no_kw['correct']}/{result_no_kw['total']}")
    else:
        print("   ⚠️ LLM 서비스 없음")
    
    # ===== 결과 요약 =====
    print("\n" + "=" * 70)
    print(f"📊 SemEval 2014 {domain.capitalize()} 결과")
    print("=" * 70)
    
    print(f"\n{'조건':<35} {'Accuracy':<15} {'정답/전체':<15}")
    print("-" * 65)
    print(f"{'A. 별점 기반 (Baseline)':<35} {result_rating['accuracy']:.2%}")
    
    if 'full_model' in results:
        r = results['full_model']
        print(f"{'B. Full Model (LLM + 키워드)':<35} {r['accuracy']:.2%}{'':<6} {r['correct']}/{r['total']}")
    
    if 'without_keyword' in results:
        r = results['without_keyword']
        print(f"{'C. w/o 키워드 추출':<35} {r['accuracy']:.2%}{'':<6} {r['correct']}/{r['total']}")
    
    # 개선율
    if 'full_model' in results:
        improvement = results['full_model']['accuracy'] - result_rating['accuracy']
        print(f"\n📈 Full Model 개선: {'+' if improvement >= 0 else ''}{improvement:.2%}")
    
    # 결과 저장
    output_path = project_root / 'experiments' / 'results' / f'semeval_{domain}.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_results = {
        'dataset': f'SemEval 2014 {domain}',
        'total_samples': len(samples),
        'llm_samples': max_samples,
        **{k: v for k, v in results.items()}
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 결과 저장: {output_path}")
    
    return results


if __name__ == "__main__":
    run_semeval_experiment(domain='laptop', max_samples=1000)

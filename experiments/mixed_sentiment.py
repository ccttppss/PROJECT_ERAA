"""
실험 2: 혼합 감정 (Multi-Aspect Sentiment) 분석 - MAMS 데이터셋 사용

목적: 한 리뷰 내에서 다른 감정을 가진 여러 aspect를 얼마나 잘 분리하는지 측정
데이터: MAMS (Multi-Aspect Multi-Sentiment) - EMNLP-IJCNLP 2019

MAMS 특징: 모든 문장이 2개 이상의 다른 감정 aspect 포함
→ 별점 기반은 이론적으로 ~33% 한계 (모든 aspect에 동일 감정 부여)
"""
import sys
import time
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Tuple
import random

# src 폴더를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
from agents.sentiment_agent import SentimentAnalysisAgent
from services.llm_service import create_llm_service
from utils.logger import get_logger
import yaml

logger = get_logger(__name__)


def load_llm_service():
    """LLM 서비스 로드"""
    config_path = project_root / 'src' / 'config' / 'llm_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        llm_config = yaml.safe_load(f)
    return create_llm_service(llm_config)


def load_mams_dataset(split: str = 'test', max_samples: int = None) -> List[Dict]:
    """
    MAMS 데이터셋 로드 (XML 파싱)
    
    Args:
        split: 'train', 'val', 'test'
        max_samples: 최대 샘플 수 (None이면 전체)
    
    Returns:
        List of {text, aspects: [{term, polarity}]}
    """
    mams_path = project_root / 'datasets' / 'mams' / f'{split}.xml'
    
    if not mams_path.exists():
        raise FileNotFoundError(f"MAMS 데이터셋 없음: {mams_path}")
    
    tree = ET.parse(mams_path)
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
            
            # MAMS polarity: positive, negative, neutral
            if term and polarity:
                aspects.append({
                    'term': term,
                    'polarity': polarity
                })
        
        if len(aspects) >= 2:  # MAMS는 2개 이상 aspect 보장
            samples.append({
                'text': text,
                'aspects': aspects
            })
    
    # 샘플 수 제한
    if max_samples and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
    
    return samples


def has_mixed_sentiment(sample: Dict) -> bool:
    """혼합 감정 (다른 polarity) 여부 확인"""
    polarities = set(a['polarity'] for a in sample['aspects'])
    return len(polarities) >= 2


def fuzzy_match_aspect(target_term: str, llm_result: Dict[str, str], nlp=None) -> str:
    """
    Fuzzy matching + spaCy 의미 유사도로 LLM 결과에서 aspect 찾기
    
    Args:
        target_term: 찾고자 하는 원본 aspect term
        llm_result: LLM 응답 {aspect: sentiment}
        nlp: spaCy 모델 (None이면 로드)
    
    Returns:
        매칭된 sentiment 또는 'not_found'
    """
    target_lower = target_term.lower().strip()
    
    # 1. 정확한 매칭 (대소문자 무시)
    for key, value in llm_result.items():
        if key.lower().strip() == target_lower:
            return value
    
    # 2. 관사 제거 후 매칭 (the, a, an)
    target_no_article = target_lower
    for article in ['the ', 'a ', 'an ']:
        target_no_article = target_no_article.replace(article, '')
    
    for key, value in llm_result.items():
        key_no_article = key.lower().strip()
        for article in ['the ', 'a ', 'an ']:
            key_no_article = key_no_article.replace(article, '')
        if key_no_article == target_no_article:
            return value
    
    # 3. 부분 일치 (target이 key에 포함되거나 key가 target에 포함)
    for key, value in llm_result.items():
        key_lower = key.lower().strip()
        if target_lower in key_lower or key_lower in target_lower:
            return value
    
    # 4. 단어 단위 부분 일치 (핵심 단어가 포함되어 있으면 매칭)
    target_words = set(target_lower.split())
    for key, value in llm_result.items():
        key_words = set(key.lower().strip().split())
        if target_words & key_words:
            return value
    
    # 5. spaCy 의미 유사도 (임계값 0.6 이상이면 매칭)
    if nlp is not None:
        target_doc = nlp(target_lower)
        best_match = None
        best_score = 0.0
        
        for key, value in llm_result.items():
            key_doc = nlp(key.lower().strip())
            if target_doc.has_vector and key_doc.has_vector:
                similarity = target_doc.similarity(key_doc)
                if similarity > best_score:
                    best_score = similarity
                    best_match = value
        
        # 유사도 0.6 이상이면 매칭
        if best_score >= 0.6:
            return best_match
    
    return 'not_found'


# spaCy 모델 글로벌 로드 (한 번만)
_nlp_model = None

def get_nlp_model():
    """spaCy 모델 싱글톤 로드"""
    global _nlp_model
    if _nlp_model is None:
        try:
            import spacy
            _nlp_model = spacy.load('en_core_web_md')
            logger.info("spaCy model loaded: en_core_web_md")
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            _nlp_model = False  # 로드 실패 표시
    return _nlp_model if _nlp_model else None


def evaluate_rating_based(samples: List[Dict]) -> Dict:
    """
    별점 기반 평가 (모든 aspect에 동일 감정 부여)
    
    가정: 혼합 감정 리뷰에서 별점 기반은 가장 빈번한 감정으로 예측
    """
    total_aspects = 0
    correct_aspects = 0
    
    for sample in samples:
        # 가장 빈번한 polarity를 예측값으로 사용 (별점 기반 가정)
        polarities = [a['polarity'] for a in sample['aspects']]
        # Baseline: 통계적으로 가장 많은 'neutral'로 모두 예측 (Majority Class)
        # MAMS 데이터셋은 중립 감정이 매우 많음
        majority_polarity = 'neutral'
        
        for aspect in sample['aspects']:
            total_aspects += 1
            if majority_polarity == aspect['polarity']:
                correct_aspects += 1
    
    return {
        'method': 'rating_based',
        'accuracy': correct_aspects / total_aspects if total_aspects > 0 else 0,
        'correct': correct_aspects,
        'total': total_aspects
    }


def evaluate_llm_based(samples: List[Dict], agent: SentimentAnalysisAgent, 
                       max_samples: int = 100) -> Dict:
    """
    LLM 기반 평가 (aspect별 감정 분리)
    """
    if not agent.llm_service:
        return {'method': 'llm_based', 'accuracy': 0, 'correct': 0, 'total': 0}
    
    # spaCy 모델 로드 (의미 유사도용)
    nlp = get_nlp_model()
    
    # 샘플 수 제한 (LLM 호출 비용)
    eval_samples = samples[:max_samples] if len(samples) > max_samples else samples
    
    total_aspects = 0
    correct_aspects = 0
    details = []
    
    for idx, sample in enumerate(eval_samples):
        text = sample['text']
        true_aspects = sample['aspects']
        
        # LLM으로 aspect 감정 분석
        try:
            # 실제 aspect term 리스트 추출
            aspect_terms = [a['term'] for a in true_aspects]
            sentiment_result = agent._llm_aspect_sentiment(text, aspect_terms)
            
            # 요청 간 딜레이 (Ollama 과부하 방지)
            time.sleep(2.0)
        except Exception as e:
            logger.warning(f"LLM 호출 실패: {e}")
            sentiment_result = {}
            time.sleep(5.0)
        
        case_result = {
            'text': text[:80] + '...' if len(text) > 80 else text,
            'aspect_results': []
        }
        
        for aspect in true_aspects:
            term = aspect['term']
            true_polarity = aspect['polarity']
            total_aspects += 1
            
            # LLM 결과에서 해당 aspect 찾기 (Fuzzy Matching + spaCy 의미 유사도)
            predicted = fuzzy_match_aspect(term, sentiment_result, nlp)
            
            # 정규화
            if predicted in ['not_mentioned', 'unknown']:
                predicted = 'not_found'
            
            is_correct = predicted == true_polarity
            if is_correct:
                correct_aspects += 1
            
            case_result['aspect_results'].append({
                'term': term,
                'true': true_polarity,
                'predicted': predicted,
                'correct': is_correct
            })
        
        details.append(case_result)
        
        # 진행률 출력
        if (idx + 1) % 20 == 0:
            print(f"   진행: {idx + 1}/{len(eval_samples)}")
    
    return {
        'method': 'llm_based',
        'accuracy': correct_aspects / total_aspects if total_aspects > 0 else 0,
        'correct': correct_aspects,
        'total': total_aspects,
        'details': details
    }


def run_mams_experiment(max_llm_samples: int = 100):
    """MAMS 혼합 감정 분석 실험 실행"""
    print("=" * 70)
    print("🔀 실험 2: 혼합 감정 분석 (MAMS Dataset)")
    print("=" * 70)
    print("📚 데이터: MAMS (EMNLP-IJCNLP 2019)")
    print("📖 특징: 모든 문장이 2+ 다른 감정 aspect 포함")
    
    # 데이터셋 로드
    print("\n📂 MAMS 데이터셋 로드...")
    try:
        samples = load_mams_dataset('test')
        mixed_count = sum(1 for s in samples if has_mixed_sentiment(s))
        total_aspects = sum(len(s['aspects']) for s in samples)
        
        print(f"   ✅ 문장 수: {len(samples)}")
        print(f"   ✅ 총 aspect 수: {total_aspects}")
        print(f"   ✅ 혼합 감정 문장: {mixed_count} ({100*mixed_count/len(samples):.1f}%)")
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
    # llm_service = None # 주석 제거
    
    config = {'log_level': 'WARNING'}
    agent = SentimentAnalysisAgent(config, llm_service)
    
    
    results = {}
    
    # 기존 결과 로드 (Baseline 결과 유지 목적)
    output_path = project_root / 'experiments' / 'results' / 'mams_mixed_sentiment.json'
    if output_path.exists():
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                if 'rating_based' in existing:
                    results['rating_based'] = existing['rating_based']
                    print("   ✅ 기존 Baseline 결과 로드됨")
        except:
            pass
    
    # ===== 방법 A: 별점 기반 =====
    print("\n" + "-" * 50)
    print("📌 방법 A: 별점 기반 (모든 aspect에 동일 감정)")
    
    # 이미 로드된 결과가 있으면 스킵
    if 'rating_based' not in results:
        result_rating = evaluate_rating_based(samples)
        results['rating_based'] = result_rating
    else:
        result_rating = results['rating_based']
    
    print(f"   Accuracy: {result_rating['accuracy']:.2%}")
    print(f"   정답: {result_rating['correct']}/{result_rating['total']}")
    
    # ===== 방법 B: LLM 기반 =====
    print("\n" + "-" * 50)
    print(f"📌 방법 B: LLM 기반 (aspect별 감정 분리, {max_llm_samples}개 샘플)")
    
    if llm_service:
        result_llm = evaluate_llm_based(samples, agent, max_samples=max_llm_samples)
        results['llm_based'] = result_llm
        
        print(f"\n   Accuracy: {result_llm['accuracy']:.2%}")
        print(f"   정답: {result_llm['correct']}/{result_llm['total']}")
        
        # 샘플 결과 출력
        if 'details' in result_llm and result_llm['details']:
            print("\n   📋 샘플 결과:")
            for detail in result_llm['details'][:3]:
                print(f"\n   \"{detail['text']}\"")
                for ar in detail['aspect_results']:
                    status = "✅" if ar['correct'] else "❌"
                    print(f"      {status} {ar['term']}: {ar['predicted']} (정답: {ar['true']})")
    else:
        print("   ⚠️ LLM 서비스 없음")
    
    # ===== 결과 요약 =====
    print("\n" + "=" * 70)
    print("📊 MAMS 혼합 감정 분석 결과")
    print("=" * 70)
    
    print(f"\n{'방법':<30} {'Accuracy':<15} {'정답/전체':<15}")
    print("-" * 60)
    print(f"{'A. 별점 기반':<30} {result_rating['accuracy']:.2%}{'':<10} {result_rating['correct']}/{result_rating['total']}")
    
    if 'llm_based' in results:
        r = results['llm_based']
        print(f"{'B. LLM 기반':<30} {r['accuracy']:.2%}{'':<10} {r['correct']}/{r['total']}")
        
        improvement = r['accuracy'] - result_rating['accuracy']
        print(f"\n📈 개선: +{improvement:.2%}")
    
    # 결과 저장
    output_path = project_root / 'experiments' / 'results' / 'mams_mixed_sentiment.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    save_results = {
        'dataset': 'MAMS',
        'total_samples': len(samples),
        'llm_samples': max_llm_samples,
        'rating_based': {k: v for k, v in results['rating_based'].items()},
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
    run_mams_experiment(max_llm_samples=1000)  # 500개 샘플로 통계적 유의성 확보

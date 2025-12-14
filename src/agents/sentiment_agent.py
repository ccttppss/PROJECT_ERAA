"""
감성 분석 에이전트 (Phase 2.5: LLM 기반 ABSA)

Phase 2 업그레이드:
- ABSA (Aspect-Based Sentiment Analysis) 구현
- 키워드 기반 aspect 추출
- aspect별 감정 분류

Phase 2.5 업그레이드:
- LLM 기반 aspect sentiment 분류 (반어법, 문맥 전환 처리)
"""
import pandas as pd
import yaml
import json
from typing import Dict, Any, List, Set, Optional
from pathlib import Path
from jinja2 import Template
from core.base_agent import BaseAgent
from services.llm_service import LLMService


class SentimentAnalysisAgent(BaseAgent):
    """
    감성 분석 에이전트 (Phase 2.5)

    기능:
    1. 별점 기반 전체 감정 분류 (Phase 1)
    2. ABSA: aspect 추출 및 aspect별 감정 분류 (Phase 2)
    3. LLM 기반 aspect sentiment 분류 (Phase 2.5) - 반어법, 문맥 전환 처리
    """

    VERSION = "2.5.0"  # Phase 2.5 - LLM-based ABSA

    def __init__(self, config: Dict[str, Any], llm_service: Optional[LLMService] = None, logger=None):
        super().__init__(config, logger)

        # LLM 서비스 (선택적)
        self.llm_service = llm_service
        
        # Aspect sentiment 프롬프트 템플릿 로드
        self.aspect_sentiment_template = self._load_aspect_sentiment_template()

        # Aspect keywords 로드
        self.aspect_keywords = self._load_aspect_keywords()
        self.logger.info(f"Loaded {len(self.aspect_keywords)} aspect categories")
        
        if self.llm_service:
            self.logger.info("LLM-based aspect sentiment analysis enabled")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        감성 분석 실행 (Phase 2)

        Args:
            input_data: {'dataframe': pd.DataFrame, ...}

        Returns:
            감성 분석 결과 + ABSA 결과
        """
        self._log_execution_start()

        df = input_data.get('dataframe')
        if df is None or not isinstance(df, pd.DataFrame):
            raise ValueError("Invalid input: 'dataframe' required")

        # Phase 1: 별점 기반 감성 분류
        sentiment_counts = self._analyze_sentiment(df)

        # 감성별 평균 리뷰 길이
        sentiment_length = self._calculate_sentiment_metrics(df)

        # Phase 2: ABSA (Aspect-Based Sentiment Analysis)
        absa_results = self.aspect_based_sentiment(df)

        result = {
            "sentiment_distribution": sentiment_counts,
            "sentiment_metrics": sentiment_length,
            "total_analyzed": len(df),
            # Phase 2 추가
            "absa": absa_results,
            "aspect_summary": self._get_aspect_summary(absa_results)
        }

        self.log_metrics("sentiment_positive", sentiment_counts.get('positive', 0))
        self.log_metrics("sentiment_negative", sentiment_counts.get('negative', 0))
        self.log_metrics("aspects_detected", len(absa_results))

        self._log_execution_end()

        return result

    def _load_aspect_sentiment_template(self) -> Optional[Template]:
        """Aspect sentiment 프롬프트 템플릿 로드"""
        template_path = Path(__file__).parent.parent / "prompts" / "aspect_sentiment.jinja2"
        
        if not template_path.exists():
            self.logger.warning(f"Aspect sentiment template not found: {template_path}")
            return None
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return Template(f.read())
        except Exception as e:
            self.logger.error(f"Failed to load aspect sentiment template: {e}")
            return None

    def _llm_aspect_sentiment(self, review_text: str, aspects: Set[str] = None) -> Dict[str, str]:
        """
        LLM을 사용하여 aspect별 감정 분류 (Aspect 지정 방식)
        
        Args:
            review_text: 리뷰 텍스트
            aspects: 분석할 aspect 집합
            
        Returns:
            {aspect_name: "positive"|"negative"|"neutral"|"not_mentioned"}
        """
        if not self.llm_service or not self.aspect_sentiment_template or not aspects:
            return {}
        
        try:
            # 프롬프트 생성 (Aspect 지정: aspect 목록과 함께 전달)
            prompt = self.aspect_sentiment_template.render(
                review_text=review_text,
                aspects=list(aspects)
            )
            
            # LLM 호출
            response = self.llm_service.generate_json(
                prompt=prompt,
                max_tokens=10000,  # 3000 → 10000 (Thinking Model 지원 강화)
                temperature=0.3  # 낮은 temperature로 일관된 결과
            )
            
            if response:
                # Flat 구조 지원: {"food": "positive", "service": "negative"}
                # Nested 구조도 지원: {"aspect_sentiments": {...}}
                if "aspect_sentiments" in response:
                    # 기존 nested 구조
                    sentiments = {}
                    for aspect, data in response["aspect_sentiments"].items():
                        if isinstance(data, dict):
                            sentiments[aspect] = data.get("sentiment", "neutral")
                        else:
                            sentiments[aspect] = str(data)
                    return sentiments
                else:
                    # 새 flat 구조: 직접 {aspect: sentiment} 형태
                    sentiments = {}
                    for aspect, sentiment in response.items():
                        if isinstance(sentiment, str) and sentiment in ["positive", "negative", "neutral", "not_mentioned"]:
                            sentiments[aspect] = sentiment
                    return sentiments
                
        except Exception as e:
            self.logger.warning(f"LLM aspect sentiment failed: {e}")
        
        return {}

    def _batch_llm_aspect_sentiment(self, df: pd.DataFrame, sample_size: int = 50) -> Dict[str, Dict[str, int]]:
        """
        배치로 LLM aspect sentiment 분석
        
        Args:
            df: 리뷰 DataFrame
            sample_size: LLM 분석할 샘플 수 (비용 최적화)
            
        Returns:
            {aspect_name: {"positive": count, "negative": count, "neutral": count}}
        """
        if not self.llm_service:
            return {}
        
        # 샘플링 (비용 최적화)
        if len(df) > sample_size:
            sample_df = df.sample(n=sample_size, random_state=42)
            self.logger.info(f"Sampling {sample_size}/{len(df)} reviews for LLM analysis")
        else:
            sample_df = df
        
        # 결과 저장
        llm_results = {aspect: {"positive": 0, "negative": 0, "neutral": 0} 
                       for aspect in self.aspect_keywords.keys()}
        
        processed = 0
        for idx, row in sample_df.iterrows():
            review_text = row.get('reviewText', '')
            if not review_text:
                continue
                
            # Aspect 추출
            aspects = self.extract_aspects(review_text)
            if not aspects:
                continue
            
            # LLM으로 감정 분류
            sentiments = self._llm_aspect_sentiment(review_text, aspects)
            
            for aspect, sentiment in sentiments.items():
                if aspect in llm_results and sentiment in ["positive", "negative", "neutral"]:
                    llm_results[aspect][sentiment] += 1
            
            processed += 1
            if processed % 10 == 0:
                self.logger.debug(f"LLM processed {processed}/{len(sample_df)} reviews")
        
        self.logger.info(f"LLM aspect sentiment completed: {processed} reviews analyzed")
        return llm_results

    def _load_aspect_keywords(self) -> Dict[str, List[str]]:
        """
        aspect_keywords YAML 파일 로드

        Returns:
            {aspect_name: [keywords]}
        """
        # YAML 파일 경로
        yaml_path = Path(__file__).parent.parent / "config" / "aspect_keywords" / "electronics.yaml"

        if not yaml_path.exists():
            self.logger.warning(f"Aspect keywords file not found: {yaml_path}")
            return {}

        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            aspects = data.get('aspects', {})

            # {aspect: [keywords]} 형태로 변환
            aspect_keywords = {}
            for aspect_name, aspect_data in aspects.items():
                keywords = aspect_data.get('keywords', [])
                aspect_keywords[aspect_name] = [kw.lower() for kw in keywords]

            return aspect_keywords

        except Exception as e:
            self.logger.error(f"Failed to load aspect keywords: {e}")
            return {}

    def extract_aspects(self, review_text: str) -> Set[str]:
        """
        리뷰 텍스트에서 언급된 aspect 추출 (키워드 매칭)

        Args:
            review_text: 리뷰 텍스트

        Returns:
            Set of aspect names mentioned in the review
        """
        if not review_text or not isinstance(review_text, str):
            return set()

        review_lower = review_text.lower()
        mentioned_aspects = set()

        for aspect_name, keywords in self.aspect_keywords.items():
            for keyword in keywords:
                if keyword in review_lower:
                    mentioned_aspects.add(aspect_name)
                    break  # 이 aspect는 발견됨, 다음 aspect로

        return mentioned_aspects

    def aspect_based_sentiment(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Aspect-Based Sentiment Analysis (ABSA) - Phase 2.5

        LLM 서비스가 있으면 LLM 기반 감정 분류 (반어법, 문맥 전환 처리)
        없으면 기존 별점 기반 폴백

        각 aspect에 대해:
        - 언급 횟수
        - 긍정/부정/중립 분포
        - 평균 별점

        Args:
            df: 리뷰 DataFrame (reviewText, overall, sentiment_label 필요)

        Returns:
            {
                aspect_name: {
                    "mention_count": int,
                    "positive": int,
                    "negative": int,
                    "neutral": int,
                    "avg_rating": float,
                    "sentiment_ratio": {"positive": %, "negative": %, ...},
                    "analysis_method": "llm" | "rating_based"  # Phase 2.5
                }
            }
        """
        if self.aspect_keywords is None or len(self.aspect_keywords) == 0:
            self.logger.warning("No aspect keywords loaded, skipping ABSA")
            return {}

        # 각 리뷰에서 aspect 추출
        df['aspects'] = df['reviewText'].apply(self.extract_aspects)

        # LLM 기반 분석 시도 (Phase 2.5)
        use_llm = self.llm_service is not None and self.aspect_sentiment_template is not None
        llm_sentiment_counts = {}
        
        if use_llm:
            self.logger.info("Using LLM-based aspect sentiment analysis (Phase 2.5)")
            llm_sentiment_counts = self._batch_llm_aspect_sentiment(df, sample_size=50)

        # 각 aspect별로 통계 수집
        absa_results = {}

        for aspect_name in self.aspect_keywords.keys():
            # 이 aspect를 언급한 리뷰 필터링
            aspect_reviews = df[df['aspects'].apply(lambda x: aspect_name in x)]

            if len(aspect_reviews) == 0:
                continue  # 언급 없으면 스킵

            total_mentions = len(aspect_reviews)
            
            # LLM 결과가 있으면 LLM 기반, 없으면 별점 기반
            if use_llm and aspect_name in llm_sentiment_counts:
                llm_counts = llm_sentiment_counts[aspect_name]
                llm_total = sum(llm_counts.values())
                
                if llm_total > 0:
                    # LLM 결과를 전체 리뷰 수에 비례하여 스케일링
                    scale_factor = total_mentions / llm_total if llm_total > 0 else 1
                    
                    sentiment_counts = {
                        'positive': int(llm_counts['positive'] * scale_factor),
                        'negative': int(llm_counts['negative'] * scale_factor),
                        'neutral': int(llm_counts['neutral'] * scale_factor)
                    }
                    analysis_method = "llm"
                else:
                    # LLM 결과가 비어있으면 별점 기반 폴백
                    sentiment_counts = aspect_reviews['sentiment_label'].value_counts().to_dict()
                    analysis_method = "rating_based"
            else:
                # 기존 별점 기반 (Phase 2 호환)
                sentiment_counts = aspect_reviews['sentiment_label'].value_counts().to_dict()
                analysis_method = "rating_based"

            absa_results[aspect_name] = {
                "mention_count": total_mentions,
                "positive": sentiment_counts.get('positive', 0),
                "negative": sentiment_counts.get('negative', 0),
                "neutral": sentiment_counts.get('neutral', 0),
                "avg_rating": float(aspect_reviews['overall'].mean()),
                "sentiment_ratio": {
                    "positive": round(sentiment_counts.get('positive', 0) / total_mentions * 100, 1),
                    "negative": round(sentiment_counts.get('negative', 0) / total_mentions * 100, 1),
                    "neutral": round(sentiment_counts.get('neutral', 0) / total_mentions * 100, 1)
                },
                "analysis_method": analysis_method  # Phase 2.5: 분석 방법 표시
            }

        method_summary = "LLM-based" if use_llm else "rating-based"
        self.logger.info(f"ABSA completed ({method_summary}): {len(absa_results)} aspects detected")

        return absa_results

    def _get_aspect_summary(self, absa_results: Dict) -> List[Dict]:
        """
        ABSA 결과를 요약 (상위 aspect만)

        Returns:
            상위 aspect 리스트 (언급 횟수 순)
        """
        if not absa_results:
            return []

        # 언급 횟수 순으로 정렬
        sorted_aspects = sorted(
            absa_results.items(),
            key=lambda x: x[1]['mention_count'],
            reverse=True
        )

        summary = []
        for aspect_name, data in sorted_aspects[:5]:  # 상위 5개만
            summary.append({
                "aspect": aspect_name,
                "mentions": data['mention_count'],
                "avg_rating": round(data['avg_rating'], 2),
                "dominant_sentiment": self._get_dominant_sentiment(data)
            })

        return summary

    def _get_dominant_sentiment(self, aspect_data: Dict) -> str:
        """aspect의 주요 감정 판단"""
        positive = aspect_data.get('positive', 0)
        negative = aspect_data.get('negative', 0)
        neutral = aspect_data.get('neutral', 0)

        max_count = max(positive, negative, neutral)

        if max_count == positive:
            return "positive"
        elif max_count == negative:
            return "negative"
        else:
            return "neutral"

    def _analyze_sentiment(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        별점 기반 감성 분류 (Phase 1 기능 유지)

        Args:
            df: 리뷰 데이터프레임

        Returns:
            감성별 카운트
        """
        if 'sentiment_label' in df.columns:
            counts = df['sentiment_label'].value_counts().to_dict()
        else:
            # sentiment_label이 없으면 overall에서 직접 계산
            counts = {
                'positive': int((df['overall'] >= 4).sum()),
                'negative': int((df['overall'] <= 2).sum()),
                'neutral': int(((df['overall'] > 2) & (df['overall'] < 4)).sum())
            }

        self.logger.info(
            "Sentiment analysis completed",
            positive=counts.get('positive', 0),
            negative=counts.get('negative', 0),
            neutral=counts.get('neutral', 0)
        )

        return counts

    def _calculate_sentiment_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        감성별 메트릭 계산 (Phase 1 기능 유지)

        Args:
            df: 리뷰 데이터프레임

        Returns:
            감성별 메트릭
        """
        metrics = {}

        if 'sentiment_label' not in df.columns:
            # sentiment_label 생성
            df['sentiment_label'] = df['overall'].apply(
                lambda x: 'positive' if x >= 4 else ('negative' if x <= 2 else 'neutral')
            )

        for sentiment in ['positive', 'negative', 'neutral']:
            sentiment_df = df[df['sentiment_label'] == sentiment]

            if len(sentiment_df) > 0:
                metrics[sentiment] = {
                    "count": len(sentiment_df),
                    "avg_length": int(sentiment_df['review_length'].mean()) if 'review_length' in sentiment_df.columns else 0,
                    "avg_rating": float(sentiment_df['overall'].mean())
                }
            else:
                metrics[sentiment] = {
                    "count": 0,
                    "avg_length": 0,
                    "avg_rating": 0.0
                }

        return metrics

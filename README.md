# E-commerce Review Analysis System (Hybrid Sequential Chain)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**규칙 기반(Rule-based) 연산**과 **단일 LLM의 추론(Reasoning)**을 결합한 **하이브리드 순차적 프롬프트 체인(Hybrid Sequential Prompt Chain)** 아키텍처입니다.

Python 라이브러리를 활용해 정밀한 통계와 감성 분석(ABSA)을 수행하고, 그 결과를 바탕으로 단일 LLM이 전체 맥락(Context)을 유지하며 **심층 분석 → 전략 수립 → 보고서 작성**의 과정을 수행합니다. 이를 통해 **할루시네이션(Hallucination)을 방지**하고 **비용 효율성**을 극대화했습니다.

> ⚠️ **데이터셋 다운로드 필요**: 이 저장소는 대용량 데이터셋 파일을 포함하지 않습니다. 아래 [데이터셋 준비](#-데이터셋-준비) 섹션을 참조하세요.

## 📊 프로젝트 개요

### 해결하는 문제

**기존 문제점:**
- **문맥 단절**: 개별 분석 도구의 파편화로 인해 데이터의 맥락이 최종 결과까지 이어지지 않음
- **신뢰성 부족**: LLM에게 모든 계산을 맡길 경우 수치 오류(Hallucination) 발생 가능성 높음
- **실행력 부재**: 분석 결과가 단순 통계에 그쳐 구체적인 비즈니스 액션으로 연결되지 않음

**제공하는 가치:**
- ✅ **뉴로-심볼릭(Neuro-Symbolic) 접근**: 정확한 연산(Python)과 고차원적 추론(LLM)의 결합
- ✅ **문맥 유지(Context Retention)**: 초기 통계 데이터가 최종 리포트까지 논리적으로 연결됨
- ✅ **비용 효율성(Cost-Efficiency)**: 단순 처리는 Python이 담당하여 LLM 토큰 비용 절감
- ✅ **ABSA 통합**: 사전 정의된 키워드 기반의 속성별 감성 분석으로 정량적 근거 마련
- ✅ **자동화된 전략 수립**: 분석된 데이터를 바탕으로 실행 가능한 Action Plan 자동 생성

## 📦 데이터셋 준비

이 저장소는 대용량 데이터셋 파일을 포함하지 않습니다. 사용하기 전에 데이터를 다운로드해야 합니다.

### Amazon Review Polarity Dataset (추천)

```bash
# datasets/ 디렉토리로 이동
cd datasets/

# 데이터셋 다운로드 (657 MB)
wget [https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz](https://s3.amazonaws.com/fast-ai-nlp/amazon_review_polarity_csv.tgz)

# 압축 해제
tar -xzf amazon_review_polarity_csv.tgz

# test.csv (168MB) 생성됨
```

### 데이터 형식

리뷰 데이터는 JSON Lines 형식이어야 합니다:

```json
{"overall": 5.0, "reviewText": "Great product!", "summary": "Excellent", "reviewTime": "01 1, 2024", "asin": "B001234567"}
{"overall": 2.0, "reviewText": "Not good", "summary": "Disappointed", "reviewTime": "01 2, 2024", "asin": "B001234567"}
```

## ✨ 주요 기능

- ✅ **Hybrid Pipeline**: Python 전처리/통계와 LLM 추론이 직렬로 연결된 파이프라인
- ✅ **Rule-based ABSA**: `electronics.yaml` 키워드 매칭을 통한 정밀한 속성 기반 감성 분석
- ✅ **Deep Insight Extraction**: 정제된 통계 데이터를 바탕으로 LLM이 심층 문제점(Pain Points) 도출
- ✅ **Logical Action Planning**: 도출된 문제점의 우선순위를 평가하고 구체적 실행 계획 수립
- ✅ **Integrated Reporting**: 전체 분석 흐름을 요약하여 경영진 의사결정용 보고서 생성
- ✅ **시각화**: Matplotlib/Seaborn을 활용한 4종 차트 자동 생성
- ✅ **웹 인터페이스**: Streamlit 대시보드를 통한 간편한 사용

## 🚀 빠른 시작

### 방법 1: 로컬 설치 + 웹 인터페이스

```bash
# 1. 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 2. 패키지 설치 (웹 인터페이스 포함)
pip install -e ".[web]"

# 3. 웹 인터페이스 실행
streamlit run scripts/web_interface.py
# 브라우저: http://localhost:8501
```

### 방법 2: CLI 명령어

```bash
# 패키지 설치
pip install -e .

# 리뷰 분석 (데이터셋 다운로드 후 실행)
python scripts/run_analysis.py --data-path datasets/test.csv --limit 100
```

### Ollama 설정 (로컬 LLM)

```bash
# Ollama 설치 ([https://ollama.com](https://ollama.com))

# gpt-oss:20b 모델 다운로드
ollama pull gpt-oss:20b

# Ollama 서버 실행
ollama serve
```

### 기본 분석 실행

```bash
python scripts/run_analysis.py \
    --data-path datasets/test.csv \
    --limit 100 \
    --output output/results.json
```

**결과**:
```
output/
├── results.json           # 전체 분석 결과 (JSON)
└── charts/                # 시각화 차트
    ├── rating_distribution.png
    ├── time_series.png
    ├── aspect_sentiment.png
    └── sentiment_pie.png
```

## 🏗️ 시스템 아키텍처

### Hybrid Sequential Chain 구조

중앙 **Orchestrator**가 데이터 흐름을 제어하며, **Python 모듈(정량 분석)**과 **단일 LLM(정성 분석)**을 순차적으로 호출하여 분석을 완성합니다.

```
┌─────────────────────────────────────────────┐
│          Review Analysis Orchestrator       │
│        (Controls the Sequential Flow)       │
└──────────────────────┬──────────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │  Phase 1: Python Processing Layer   │
    │  (Accuracy & Statistics Focus)      │
    │                                     │
    │  1. Data Preprocessor (Cleaning)    │
    │  2. Data Collection (Basic Stats)   │
    │  3. Sentiment Agent (ABSA/Rule)     │
    └──────────────────┬──────────────────┘
                       │ (Structured Context)
    ┌──────────────────▼──────────────────┐
    │  Phase 2: LLM Reasoning Layer       │
    │  (Single LLM - Chain of Thought)    │
    │                                     │
    │  4. Insight Agent (Analyst Role)    │
    │     -> Extract Pain Points/Strengths│
    │                                     │
    │  5. Action Agent (Strategist Role)  │
    │     -> Plan Quick Wins/Long-term    │
    │                                     │
    │  6. Report Agent (Reporter Role)    │
    │     -> Generate Executive Summary   │
    └─────────────────────────────────────┘
```

### 데이터 처리 흐름 (Process Flow)

1.  **Data Processing (Python)**: HTML 제거, 비정상 데이터 필터링, 텍스트 정규화.
2.  **Stats & ABSA (Python)**: 평점 분포 계산 및 키워드 매칭을 통한 속성별(배터리, 가격 등) 감성 분류.
3.  **Insight Extraction (LLM)**: 위에서 산출된 통계 데이터를 바탕으로 LLM이 '분석가'가 되어 근본 원인 분석.
4.  **Action Planning (LLM)**: 도출된 인사이트를 바탕으로 LLM이 '전략가'가 되어 우선순위별 실행 계획 수립.
5.  **Report Generation (LLM)**: LLM이 '보고자'가 되어 전체 내용을 종합한 비즈니스 리포트 작성.

## 📁 프로젝트 구조

```
agentic_ai/
├── src/                     ⭐ 모든 소스 코드
│   ├── agents/              # 분석 모듈 (Python 로직 및 LLM 프롬프트 래퍼)
│   │   ├── data_collection_agent.py # 기본 통계 (Python)
│   │   ├── sentiment_agent.py       # ABSA 분석 (Rule-based Python)
│   │   ├── insight_agent.py         # 인사이트 도출 (LLM)
│   │   ├── action_planning_agent.py # 실행 계획 (LLM)
│   │   └── report_agent.py          # 리포트 생성 (LLM)
│   ├── core/                # 오케스트레이터 및 기본 클래스
│   ├── services/            # LLM 서비스(Ollama/Claude), 시각화
│   ├── utils/               # 로거, 에러 핸들러
│   ├── data/                # 데이터 로더 및 전처리
│   ├── prompts/             # 단계별 프롬프트 템플릿 (Jinja2)
│   └── config/              # 설정 파일 (llm_config.yaml)
│
├── datasets/                # 데이터셋 (다운로드 필요)
├── scripts/                 # 실행 스크립트
├── README.md                # 프로젝트 문서
└── requirements.txt         # 의존성 목록
```

## 📖 사용법

### Python

```python
from core.orchestrator import ReviewAnalysisOrchestrator

# Orchestrator 초기화
orchestrator = ReviewAnalysisOrchestrator(
    config={'output_dir': 'output'},
    llm_config_path='src/config/llm_config.yaml'
)

# 분석 실행 (Hybrid Pipeline)
# Python 전처리 -> ABSA -> LLM 추론 순으로 자동 실행
result = orchestrator.run_analysis(
    data_path='datasets/test.csv',
    product_id='B0123456',
    limit=500,
    enable_llm=True
)

# 결과 확인
print(f"Total Reviews: {result['basic_stats']['total_reviews']}")
print(f"Insights: {len(result['insights']['pain_points'])} identified")
print(f"Actions: {len(result['action_plan']['quick_wins'])} generated")
```

## ⚡ 성능 최적화 전략

- **Hybrid Processing**: 수치 계산이 필요한 통계/ABSA는 Python으로 빠르게 처리하고, 고차원 추론이 필요한 부분만 LLM을 사용하여 속도와 정확도 동시에 확보.
- **Context Optimization**: 각 단계별로 핵심 정보만 요약하여 다음 프롬프트로 전달함으로써 LLM 토큰 비용 절감.
- **LLM Caching**: 동일한 프롬프트에 대한 응답을 캐싱하여 반복 실험 시 처리 속도 향상.

## 🛠️ 기술 스택

- **Python 3.9+**
- **pandas**: 데이터 전처리 및 정량 분석
- **Jinja2**: 프롬프트 템플릿 관리
- **Ollama**: 로컬 LLM (gpt-oss:20b 등) 추론 엔진
- **Matplotlib/Seaborn**: 데이터 시각화

## 📝 라이선스

MIT License - 자유롭게 사용, 수정, 배포할 수 있습니다.

---

**Version**: 4.0.0 (Hybrid Neuro-Symbolic Architecture)
**Last Updated**: 2025-10-20
**Status**: Production Ready ✅

"""
Streamlit 웹 인터페이스 (Phase 5)

간단한 웹 대시보드로 리뷰 분석 시스템을 사용할 수 있습니다.

실행:
    streamlit run scripts/web_interface.py
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import sys
from datetime import datetime
import time

# src 폴더를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from core.orchestrator import ReviewAnalysisOrchestrator
from utils.logger import get_logger

logger = get_logger(__name__)


# 페이지 설정
st.set_page_config(
    page_title="Amazon Review Analysis System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """메인 함수"""

    # 헤더
    st.markdown('<p class="main-header">📊 Amazon Review Analysis System</p>', unsafe_allow_html=True)
    st.markdown("**Version 4.0.0** | Multi-Agent AI System for Business Insights")
    st.divider()

    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")

        # 데이터 파일 업로드
        st.subheader("1. 데이터 업로드")

        # 샘플 데이터 버튼 추가
        use_sample = st.button("🎯 샘플 데이터로 바로 테스트", use_container_width=True)

        if use_sample:
            st.session_state['use_sample'] = True

        uploaded_file = st.file_uploader(
            "Amazon 리뷰 JSON 파일",
            type=['json'],
            help="Amazon 리뷰 데이터 JSON 파일을 업로드하세요"
        )

        if uploaded_file:
            st.session_state['use_sample'] = False

        # 분석 옵션
        st.subheader("2. 분석 옵션")

        product_id = st.text_input(
            "제품 ID (ASIN)",
            value="ALL",
            help="분석할 제품의 ASIN 코드 (ALL = 모든 리뷰 분석)",
            placeholder="예: B000000000, B001234567, B007XYZ123..."
        )

        st.caption("💡 **ALL** 입력 시 데이터셋의 모든 리뷰를 분석합니다")

        limit = st.slider(
            "분석할 리뷰 수",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="설정한 개수만큼 LLM이 분석합니다 (무작위 섞기)"
        )

        st.caption("💡 처리 시간: 10개=30초, 50개=2분, 100개=3-4분, 500개=10-15분, 1000개=20-30분")
        st.caption("✨ **자동 배치 처리**: 100개 초과 시 자동으로 배치 분할하여 안정적으로 처리합니다")

        enable_llm = st.checkbox(
            "LLM 분석 활성화",
            value=True,
            help="인사이트 추출 및 실행 계획 생성 (시간 소요)"
        )

        cache_enabled = st.checkbox(
            "캐싱 활성화",
            value=True,
            help="LLM 응답 캐싱으로 속도 향상"
        )

        st.divider()

        # 정보
        st.subheader("ℹ️ 시스템 정보")
        st.info("""
        **Phase 1-5 완료**
        - 감성 분석 (ABSA)
        - LLM 인사이트 추출
        - 실행 계획 생성
        - 시각화
        - 경쟁사 비교
        """)

    # 메인 영역
    if uploaded_file is None and not st.session_state.get('use_sample', False):
        # 시작 화면
        st.info("👈 사이드바에서 '샘플 데이터로 바로 테스트' 버튼을 클릭하거나 리뷰 데이터 파일을 업로드하세요!")

        # 기능 소개
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("📈 기본 분석")
            st.write("- 평점 분포")
            st.write("- 시간 추이")
            st.write("- 감성 분류")

        with col2:
            st.subheader("🔍 ABSA 분석")
            st.write("- Aspect 추출")
            st.write("- Aspect별 감성")
            st.write("- 문제점 발견")

        with col3:
            st.subheader("💡 LLM 인사이트")
            st.write("- 문제점 도출")
            st.write("- 실행 계획")
            st.write("- 경영진 보고서")

        # 샘플 결과 표시
        st.subheader("📊 샘플 분석 결과")

        sample_data = {
            "총 리뷰 수": 100,
            "평균 평점": 3.8,
            "긍정 리뷰": "45%",
            "부정 리뷰": "30%"
        }

        cols = st.columns(4)
        for i, (key, value) in enumerate(sample_data.items()):
            with cols[i]:
                st.metric(label=key, value=value)

        return

    # 분석 실행
    st.subheader("🚀 분석 실행")

    if st.button("분석 시작", type="primary", use_container_width=True):
        # 진행 상황
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # 1. 파일 저장
            status_text.text("📁 파일 준비 중...")
            progress_bar.progress(10)

            # 샘플 데이터 사용 여부 확인
            if st.session_state.get('use_sample', False):
                # 원본 CSV에서 무작위 샘플링 (매번 다른 리뷰 선택)
                import csv
                import json
                import random
                import hashlib

                source_file = Path("datasets/test.csv")

                # 1단계: CSV 파일 전체 라인 수 카운트 (빠른 샘플링을 위해)
                with open(source_file, 'r', encoding='utf-8') as f:
                    total_lines = sum(1 for _ in f)

                # 2단계: 무작위로 샘플링할 라인 번호 선택
                sample_size = min(limit, total_lines)
                sampled_lines = sorted(random.sample(range(total_lines), sample_size))

                # 3단계: 선택된 라인만 읽어서 JSON으로 변환
                sample_reviews = []
                with open(source_file, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    for line_num, row in enumerate(csv_reader):
                        if line_num in sampled_lines:
                            try:
                                label = row[0].strip('"')
                                title = row[1].strip('"') if len(row) > 1 else ""
                                review_text = row[2].strip('"') if len(row) > 2 else ""

                                # CSV 라벨을 별점으로 변환 (1=부정, 2=긍정)
                                overall = 1.0 if label == "1" else 5.0

                                # 고유 ID 생성
                                review_hash = hashlib.md5(review_text.encode()).hexdigest()[:16]

                                # Amazon 리뷰 JSON 형식
                                review_json = {
                                    "reviewerID": f"R{review_hash}",
                                    "asin": "B000000000",
                                    "reviewerName": f"Reviewer_{len(sample_reviews)+1}",
                                    "helpful": [0, 0],
                                    "reviewText": review_text,
                                    "overall": overall,
                                    "summary": title,
                                    "unixReviewTime": 1577836800,
                                    "reviewTime": "01 01, 2020"
                                }
                                sample_reviews.append(review_json)
                            except:
                                continue

                        if len(sample_reviews) >= sample_size:
                            break

                # 4단계: 임시 파일로 저장
                temp_path = Path(f"output/temp/sample_{limit}_reviews.json")
                temp_path.parent.mkdir(parents=True, exist_ok=True)

                with open(temp_path, 'w', encoding='utf-8') as f:
                    for review in sample_reviews:
                        f.write(json.dumps(review, ensure_ascii=False) + '\n')

                st.success(f"✅ 원본 CSV에서 {len(sample_reviews)}개 리뷰를 무작위로 추출했습니다! (매번 다른 리뷰)")
            else:
                # 업로드된 파일 사용
                temp_path = Path("output/temp") / uploaded_file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)

                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

            # 2. Orchestrator 초기화
            status_text.text("⚙️ 시스템 초기화 중...")
            progress_bar.progress(20)

            config = {
                'output_dir': 'output/web',
                'cache_enabled': cache_enabled,
                'aspect_keywords_path': 'src/config/aspect_keywords/electronics.yaml'
            }

            orchestrator = ReviewAnalysisOrchestrator(
                config=config,
                llm_config_path='src/config/llm_config.yaml'
            )

            # 3. 분석 실행
            status_text.text("🔬 분석 실행 중...")
            progress_bar.progress(30)

            start_time = time.time()

            # product_id가 "ALL", "UNKNOWN" 등이면 None으로 변경 (모든 리뷰 분석)
            actual_product_id = None if product_id.upper() in ["ALL", "UNKNOWN"] else product_id

            result = orchestrator.run_analysis(
                data_path=str(temp_path),
                product_id=actual_product_id,
                limit=limit,
                enable_llm=enable_llm
            )

            execution_time = time.time() - start_time

            progress_bar.progress(100)
            status_text.text("✅ 분석 완료!")

            # 4. 결과 표시
            st.success(f"분석 완료! (소요 시간: {execution_time:.2f}초)")

            # 기본 통계
            st.subheader("📊 기본 통계")

            basic_stats = result['basic_stats']

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    label="총 리뷰 수",
                    value=basic_stats['total_reviews']
                )

            with col2:
                st.metric(
                    label="평균 평점",
                    value=f"{basic_stats['avg_rating']:.2f}",
                    delta=f"{basic_stats['avg_rating'] - 3.0:.2f}"
                )

            with col3:
                sentiment_dist = result['sentiment_analysis']['sentiment_distribution']
                total = sum(sentiment_dist.values())
                positive_ratio = sentiment_dist.get('positive', 0) / total * 100 if total > 0 else 0
                st.metric(
                    label="긍정 비율",
                    value=f"{positive_ratio:.1f}%"
                )

            with col4:
                negative_ratio = sentiment_dist.get('negative', 0) / total * 100 if total > 0 else 0
                st.metric(
                    label="부정 비율",
                    value=f"{negative_ratio:.1f}%"
                )

            # 평점 분포
            st.subheader("⭐ 평점 분포")

            rating_dist = basic_stats['rating_distribution']
            rating_df = pd.DataFrame({
                '평점': list(rating_dist.keys()),
                '리뷰 수': list(rating_dist.values())
            })

            st.bar_chart(rating_df.set_index('평점'))

            # 감성 분석
            st.subheader("😊 감성 분석")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**감성 분포**")
                sentiment_df = pd.DataFrame({
                    '감성': list(sentiment_dist.keys()),
                    '개수': list(sentiment_dist.values())
                })
                st.dataframe(sentiment_df, use_container_width=True)

            with col2:
                st.write("**Aspect 요약 (Top 5)**")
                aspect_summary = result['sentiment_analysis'].get('aspect_summary', [])[:5]

                if aspect_summary:
                    aspect_df = pd.DataFrame(aspect_summary)
                    # 컬럼 이름 확인: 'mentions' (sentiment_agent.py line 216)
                    if 'mentions' in aspect_df.columns:
                        st.dataframe(aspect_df[['aspect', 'mentions', 'avg_rating', 'dominant_sentiment']], use_container_width=True)
                    else:
                        st.dataframe(aspect_df, use_container_width=True)
                else:
                    st.info("Aspect 데이터가 없습니다.")

            # LLM 인사이트 (활성화된 경우)
            if enable_llm and 'insights' in result and result['insights']:
                st.subheader("💡 LLM 인사이트")

                # 탭으로 구분: 강점 먼저, 문제점은 뒤로
                tab1, tab2 = st.tabs(["✅ 강점 (Strengths)", "❌ 문제점 (Pain Points)"])

                with tab1:
                    strengths = result['insights'].get('strengths', [])

                    if strengths:
                        st.write(f"**발견된 강점: {len(strengths)}개**")

                        for i, strength in enumerate(strengths[:5], 1):
                            with st.expander(f"강점 {i}: {strength.get('feature', 'N/A')}"):
                                st.write(f"**빈도**: {strength.get('frequency', 'N/A')}")

                                quotes = strength.get('representative_quotes', [])
                                if quotes:
                                    st.write("**대표 인용**:")
                                    for quote in quotes[:3]:
                                        st.success(quote)
                    else:
                        st.info("강점이 발견되지 않았습니다.")

                with tab2:
                    pain_points = result['insights'].get('pain_points', [])

                    if pain_points:
                        st.write(f"**발견된 문제점: {len(pain_points)}개**")

                        for i, pain_point in enumerate(pain_points[:5], 1):
                            with st.expander(f"문제점 {i}: {pain_point.get('issue', 'N/A')}"):
                                st.write(f"**빈도**: {pain_point.get('frequency', 'N/A')}")
                                st.write(f"**심각도**: {pain_point.get('severity', 'N/A')}")

                                quotes = pain_point.get('representative_quotes', [])
                                if quotes:
                                    st.write("**대표 인용**:")
                                    for quote in quotes[:3]:
                                        st.info(quote)
                    else:
                        st.info("문제점이 발견되지 않았습니다.")

            # 실행 계획 (활성화된 경우)
            if enable_llm and 'action_plan' in result and result['action_plan']:
                st.subheader("🎯 실행 계획")

                action_plan = result['action_plan'].get('action_plan', {})

                tab1, tab2, tab3 = st.tabs(["Quick Wins", "Medium-term", "Long-term"])

                with tab1:
                    quick_wins = action_plan.get('quick_wins', [])
                    if quick_wins:
                        for i, action in enumerate(quick_wins, 1):
                            st.markdown(f"**{i}. {action.get('action', 'N/A')}**")
                            st.write(f"- 근거: {action.get('rationale', 'N/A')}")
                            st.write(f"- 예상 효과: {action.get('expected_impact', 'N/A')}")
                            st.divider()
                    else:
                        st.info("Quick Wins 없음")

                with tab2:
                    medium_actions = action_plan.get('medium_term_actions', [])
                    if medium_actions:
                        for i, action in enumerate(medium_actions, 1):
                            st.markdown(f"**{i}. {action.get('action', 'N/A')}**")
                            st.write(f"- 근거: {action.get('rationale', 'N/A')}")
                            st.divider()
                    else:
                        st.info("Medium-term Actions 없음")

                with tab3:
                    long_actions = action_plan.get('long_term_actions', [])
                    if long_actions:
                        for i, action in enumerate(long_actions, 1):
                            st.markdown(f"**{i}. {action.get('action', 'N/A')}**")
                            st.write(f"- 근거: {action.get('rationale', 'N/A')}")
                            st.divider()
                    else:
                        st.info("Long-term Actions 없음")

            # 시각화 (있는 경우)
            if 'visualizations' in result and result['visualizations']:
                st.subheader("📈 시각화")

                charts = result['visualizations'].get('charts', {})

                if charts:
                    col1, col2 = st.columns(2)

                    chart_items = list(charts.items())

                    for i, (name, path) in enumerate(chart_items):
                        if Path(path).exists():
                            with col1 if i % 2 == 0 else col2:
                                st.image(path, caption=name, use_container_width=True)

            # JSON 다운로드
            st.subheader("💾 결과 다운로드")

            json_str = json.dumps(result, indent=2, ensure_ascii=False, default=str)

            st.download_button(
                label="📥 JSON 결과 다운로드",
                data=json_str,
                file_name=f"analysis_result_{product_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"❌ 에러 발생: {str(e)}")

            # LLM 에러인 경우 도움말 표시
            if "LLM" in str(e) or "None response" in str(e):
                st.warning("""
                **LLM 에러 해결 방법:**
                1. **리뷰 수 줄이기**: 10-20개로 설정
                2. **잠시 후 재시도**: Ollama 서버가 바쁠 수 있습니다
                3. **캐싱 활성화**: 응답 속도 향상
                """)

            # 자세한 에러 정보 (확장 가능)
            with st.expander("🔍 상세 에러 정보"):
                st.code(str(e))

            logger.error(f"Web interface error: {e}", exc_info=True)
            progress_bar.empty()
            status_text.empty()


if __name__ == '__main__':
    main()

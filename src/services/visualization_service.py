"""
시각화 서비스 (Phase 3)

기능:
1. 평점 분포 차트 (histogram)
2. 시간 추이 차트 (line plot)
3. Aspect별 감정 분포 차트 (stacked bar)
4. 전체 감정 파이 차트
"""

import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server environments

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging


class VisualizationService:
    """
    리뷰 분석 결과 시각화 서비스 (Phase 3)

    차트 생성:
    - 평점 분포
    - 시간 추이
    - Aspect별 감정 분포
    - 전체 감정 비율
    """

    VERSION = "3.0.0"

    def __init__(self, output_dir: str = "output/charts", logger: Optional[logging.Logger] = None):
        """
        Args:
            output_dir: 차트 저장 디렉토리
            logger: 로거 인스턴스
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logger or self._get_default_logger()

        # Seaborn 스타일 설정
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        self.logger.info(f"VisualizationService initialized (v{self.VERSION}) | output_dir={self.output_dir}")

    def _get_default_logger(self):
        """기본 로거 생성"""
        logger = logging.getLogger("VisualizationService")
        logger.setLevel(logging.INFO)
        return logger

    def plot_rating_distribution(
        self,
        df: pd.DataFrame,
        filename: str = "rating_distribution.png"
    ) -> Path:
        """
        평점 분포 차트 생성 (histogram)

        Args:
            df: 리뷰 DataFrame (overall 컬럼 필요)
            filename: 저장 파일명

        Returns:
            저장된 파일 경로
        """
        self.logger.info("Generating rating distribution chart...")

        fig, ax = plt.subplots(figsize=(10, 6))

        # 평점별 카운트
        rating_counts = df['overall'].value_counts().sort_index()

        # Histogram
        ax.bar(rating_counts.index, rating_counts.values, color='skyblue', edgecolor='black')

        # 레이블 및 제목
        ax.set_xlabel('Rating (Stars)', fontsize=12)
        ax.set_ylabel('Number of Reviews', fontsize=12)
        ax.set_title('Rating Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks([1, 2, 3, 4, 5])

        # 값 표시
        for i, v in zip(rating_counts.index, rating_counts.values):
            ax.text(i, v + max(rating_counts.values) * 0.02, str(v),
                   ha='center', va='bottom', fontsize=10)

        # 통계 정보 추가
        avg_rating = df['overall'].mean()
        total_reviews = len(df)

        stats_text = f"Total: {total_reviews} reviews\nAverage: {avg_rating:.2f}/5.0"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 저장
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Chart saved: {output_path}")
        return output_path

    def plot_time_series(
        self,
        df: pd.DataFrame,
        filename: str = "time_series.png"
    ) -> Path:
        """
        시간 추이 차트 생성 (line plot)

        Args:
            df: 리뷰 DataFrame (date, overall 컬럼 필요)
            filename: 저장 파일명

        Returns:
            저장된 파일 경로
        """
        self.logger.info("Generating time series chart...")

        # date 컬럼이 없으면 스킵
        if 'date' not in df.columns:
            self.logger.warning("No 'date' column found, skipping time series chart")
            return None

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 데이터 준비
        df_sorted = df.sort_values('date').copy()
        df_sorted['year_month'] = df_sorted['date'].dt.to_period('M')

        # 월별 평균 평점
        monthly_avg = df_sorted.groupby('year_month')['overall'].mean()
        monthly_count = df_sorted.groupby('year_month').size()

        # 차트 1: 평균 평점 추이
        ax1.plot(monthly_avg.index.astype(str), monthly_avg.values,
                marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax1.set_xlabel('Month', fontsize=12)
        ax1.set_ylabel('Average Rating', fontsize=12)
        ax1.set_title('Average Rating Over Time', fontsize=14, fontweight='bold')
        ax1.axhline(y=monthly_avg.mean(), color='red', linestyle='--',
                   alpha=0.5, label=f'Overall Avg: {monthly_avg.mean():.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # x축 레이블 회전
        ax1.tick_params(axis='x', rotation=45)

        # 차트 2: 리뷰 개수 추이
        ax2.bar(monthly_count.index.astype(str), monthly_count.values,
               color='#A23B72', alpha=0.7)
        ax2.set_xlabel('Month', fontsize=12)
        ax2.set_ylabel('Number of Reviews', fontsize=12)
        ax2.set_title('Review Volume Over Time', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # x축 레이블 회전
        ax2.tick_params(axis='x', rotation=45)

        plt.tight_layout()

        # 저장
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Chart saved: {output_path}")
        return output_path

    def plot_aspect_sentiment(
        self,
        absa_results: Dict[str, Dict[str, Any]],
        filename: str = "aspect_sentiment.png",
        top_n: int = 10
    ) -> Path:
        """
        Aspect별 감정 분포 차트 (stacked bar chart)

        Args:
            absa_results: ABSA 분석 결과 (sentiment_agent 출력)
            filename: 저장 파일명
            top_n: 상위 N개 aspect만 표시

        Returns:
            저장된 파일 경로
        """
        self.logger.info("Generating aspect sentiment chart...")

        if not absa_results:
            self.logger.warning("No ABSA results, skipping aspect sentiment chart")
            return None

        # 데이터 준비
        aspects = []
        positive_counts = []
        negative_counts = []
        neutral_counts = []

        # 언급 횟수 순으로 정렬
        sorted_aspects = sorted(
            absa_results.items(),
            key=lambda x: x[1]['mention_count'],
            reverse=True
        )[:top_n]

        for aspect_name, data in sorted_aspects:
            aspects.append(aspect_name.replace('_', ' ').title())
            positive_counts.append(data['positive'])
            negative_counts.append(data['negative'])
            neutral_counts.append(data['neutral'])

        # 차트 생성
        fig, ax = plt.subplots(figsize=(12, 8))

        x = range(len(aspects))
        width = 0.6

        # Stacked bar chart
        p1 = ax.barh(x, positive_counts, width, label='Positive', color='#2ECC71')
        p2 = ax.barh(x, negative_counts, width, left=positive_counts,
                    label='Negative', color='#E74C3C')
        p3 = ax.barh(x, neutral_counts, width,
                    left=[p+n for p, n in zip(positive_counts, negative_counts)],
                    label='Neutral', color='#95A5A6')

        # 레이블
        ax.set_xlabel('Number of Mentions', fontsize=12)
        ax.set_ylabel('Aspect', fontsize=12)
        ax.set_title(f'Sentiment Distribution by Aspect (Top {len(aspects)})',
                    fontsize=14, fontweight='bold')
        ax.set_yticks(x)
        ax.set_yticklabels(aspects)
        ax.legend(loc='lower right')

        # 그리드
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()

        # 저장
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Chart saved: {output_path}")
        return output_path

    def plot_sentiment_pie(
        self,
        sentiment_distribution: Dict[str, int],
        filename: str = "sentiment_pie.png"
    ) -> Path:
        """
        전체 감정 분포 파이 차트

        Args:
            sentiment_distribution: 감정 분포 {'positive': N, 'negative': M, 'neutral': K}
            filename: 저장 파일명

        Returns:
            저장된 파일 경로
        """
        self.logger.info("Generating sentiment pie chart...")

        fig, ax = plt.subplots(figsize=(8, 8))

        # 데이터 준비
        labels = []
        sizes = []
        colors = []
        explode = []

        sentiment_map = {
            'positive': ('Positive', '#2ECC71', 0.05),
            'negative': ('Negative', '#E74C3C', 0.05),
            'neutral': ('Neutral', '#95A5A6', 0)
        }

        for sentiment, count in sentiment_distribution.items():
            if count > 0:
                label, color, exp = sentiment_map.get(sentiment, (sentiment.title(), '#3498DB', 0))
                labels.append(f'{label}\n({count})')
                sizes.append(count)
                colors.append(color)
                explode.append(exp)

        # 파이 차트
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            explode=explode,
            autopct='%1.1f%%',
            shadow=True,
            startangle=90,
            textprops={'fontsize': 12}
        )

        # 퍼센트 텍스트 스타일
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_weight('bold')
            autotext.set_fontsize(11)

        ax.set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold', pad=20)

        # 동일한 비율로 원형 유지
        ax.axis('equal')

        plt.tight_layout()

        # 저장
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Chart saved: {output_path}")
        return output_path

    def generate_all_charts(
        self,
        df: pd.DataFrame,
        sentiment_distribution: Dict[str, int],
        absa_results: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> Dict[str, Path]:
        """
        모든 차트 일괄 생성

        Args:
            df: 리뷰 DataFrame
            sentiment_distribution: 전체 감정 분포
            absa_results: ABSA 결과 (선택)

        Returns:
            생성된 차트 파일 경로 딕셔너리
        """
        self.logger.info("Generating all charts...")

        charts = {}

        # 1. 평점 분포
        charts['rating_distribution'] = self.plot_rating_distribution(df)

        # 2. 시간 추이
        time_series_path = self.plot_time_series(df)
        if time_series_path:
            charts['time_series'] = time_series_path

        # 3. 전체 감정 파이 차트
        charts['sentiment_pie'] = self.plot_sentiment_pie(sentiment_distribution)

        # 4. Aspect별 감정 (ABSA 있을 경우)
        if absa_results:
            aspect_path = self.plot_aspect_sentiment(absa_results)
            if aspect_path:
                charts['aspect_sentiment'] = aspect_path

        self.logger.info(f"Generated {len(charts)} charts")

        return charts

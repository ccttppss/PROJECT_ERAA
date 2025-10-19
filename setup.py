"""
Amazon Review Analysis System - Setup Configuration

Phase 5: Packaging and Deployment
"""
from setuptools import setup, find_packages
from pathlib import Path

# 프로젝트 루트 디렉토리
HERE = Path(__file__).parent

# README 파일 읽기
README = (HERE / "README.md").read_text(encoding='utf-8')

# requirements.txt에서 의존성 읽기
with open(HERE / "requirements.txt", encoding='utf-8') as f:
    REQUIREMENTS = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith('#')
    ]

# 버전 정보
VERSION = "4.0.0"

setup(
    name="amazon-review-analysis",
    version=VERSION,
    description="Multi-Agent AI system for Amazon review analysis with actionable business insights",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Claude Code",
    author_email="noreply@anthropic.com",
    url="https://github.com/anthropics/claude-code",
    license="MIT",

    # 패키지 설정 (src/ 폴더 사용)
    packages=find_packages(where='src', exclude=["tests", "tests.*"]),
    package_dir={'': 'src'},
    include_package_data=True,

    # Python 버전 요구사항
    python_requires=">=3.9",

    # 의존성
    install_requires=REQUIREMENTS,

    # 선택적 의존성
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'web': [
            'streamlit>=1.28.0',
            'plotly>=5.17.0',
        ],
        'all': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'streamlit>=1.28.0',
            'plotly>=5.17.0',
        ]
    },

    # 콘솔 스크립트 (CLI 명령어)
    entry_points={
        'console_scripts': [
            'review-analysis=scripts.run_analysis:main',
            'review-evaluate=scripts.evaluate_system:main',
            'review-experiment=scripts.run_experiments:main',
        ],
    },

    # 분류자 (PyPI)
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],

    # 키워드
    keywords=[
        "amazon",
        "reviews",
        "sentiment-analysis",
        "nlp",
        "llm",
        "multi-agent",
        "business-intelligence",
        "absa",
        "aspect-based-sentiment-analysis",
    ],

    # 프로젝트 URL
    project_urls={
        "Documentation": "https://github.com/anthropics/claude-code/tree/main/docs",
        "Source": "https://github.com/anthropics/claude-code",
        "Tracker": "https://github.com/anthropics/claude-code/issues",
    },

    # 패키지 데이터
    package_data={
        'config': ['*.yaml', 'aspect_keywords/*.yaml'],
        'prompts': ['*.jinja2'],
    },

    # 데이터 파일
    data_files=[
        ('config', ['src/config/llm_config.yaml']),
    ],

    # Zip 안전 여부
    zip_safe=False,
)

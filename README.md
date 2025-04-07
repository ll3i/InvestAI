# 투자 챗봇 프로젝트

이 프로젝트는 OpenAI의 GPT 모델을 활용한 투자 자문 챗봇입니다.

## 주요 기능

- 사용자 맞춤형 투자 조언 제공
- 포트폴리오 분석 및 예측
- 다중 AI 에이전트 시스템을 통한 종합적인 투자 의사결정 지원

## 설치 방법

1. 저장소 클론
```bash
git clone [repository-url]
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. 환경 설정
- OpenAI API 키 설정
- 필요한 설정 파일 구성

## 환경 설정

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. API 키 설정:
- `.env` 파일을 생성하고 다음 내용을 추가합니다:
```
OPENAI_API_KEY=your_api_key_here
```
- 또는 환경 변수로 설정:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## 사용 방법

1. 프로그램 실행
```bash
python src/minerva.py
```

2. GUI 인터페이스를 통해 투자 관련 질문 입력
3. AI 에이전트들의 분석 결과 확인
4. 포트폴리오 예측 결과 확인

## 프로젝트 구조

```
.
├── src/                    # 소스 코드
├── docs/                   # 문서
├── prompts/               # AI 프롬프트 템플릿
├── config/                # 설정 파일
├── data/                  # 데이터 파일
└── requirements.txt       # 의존성 패키지 목록
```

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다. 
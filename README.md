# 🚗 LangGraph 기반 자동차 에이전트 - Streamlit UI

LangGraph를 사용하여 구현된 자동차 에이전트의 Streamlit 웹 인터페이스입니다.

## 🌟 주요 기능

### 1. **자동차 에이전트 채팅**
- **자동차 제어**: 창문, 조명, 오디오, 온도 등 차량 기능 제어
- **매뉴얼 검색**: Corrective RAG를 활용한 자동차 매뉴얼 검색
- **일반 대화**: 친근한 AI 어시스턴트와의 대화

### 2. **PDF 문서 처리**
- 자동차 매뉴얼 PDF 업로드 및 구조화
- 향상된 PDF 처리기를 통한 섹션별 분석
- 벡터 데이터베이스 자동 생성

### 3. **시스템 모니터링**
- 벡터 DB 상태 및 통계 표시
- 처리 진행 상황 실시간 모니터링
- 채팅 기록 및 성능 분석

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env` 파일을 생성하고 OpenAI API 키를 설정하세요:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Streamlit 앱 실행
```bash
# 방법 1: 직접 실행
streamlit run streamlit_car_agent.py

# 방법 2: 실행 스크립트 사용
python run_streamlit.py
```

### 4. 웹 브라우저에서 접속
- 기본 URL: `http://localhost:8501`
- 포트 변경: `streamlit run streamlit_car_agent.py --server.port 8502`

## 📱 사용법

### 1. **채팅 탭**
- 자동차 에이전트와 실시간 대화
- 자동차 제어, 매뉴얼 검색, 일반 대화 지원
- 채팅 기록 자동 저장

**예시 질문:**
- "창문을 열어줘" (자동차 제어)
- "엔진 시동 방법을 알려줘" (매뉴얼 검색)
- "안녕하세요" (일반 대화)

### 2. **PDF 처리 탭**
- 자동차 매뉴얼 PDF 파일 업로드
- 자동 구조화 및 벡터 DB 생성
- 처리 결과 통계 확인

### 3. **시스템 정보 탭**
- 벡터 DB 상태 및 문서 수 확인
- 구조 분석 결과 조회
- 채팅 기록 통계

## 🏗️ 아키텍처

### **LangGraph 기반 에이전트 구조**
```
사용자 입력
    ↓
라우팅 시스템
    ↓
┌─────────────┬─────────────┬─────────────┐
│ 자동차 제어  │ 매뉴얼 검색  │ 일반 대화    │
│             │ (Corrective │             │
│             │    RAG)     │             │
└─────────────┴─────────────┴─────────────┘
    ↓
최종 응답
```

### **Corrective RAG 프로세스**
```
질문 분석 → 컨텍스트 검색 → 컨텍스트 개선 → 답변 생성 → 답변 검증
    ↑                                                      ↓
    └─────────────── 반복 개선 (최대 3회) ──────────────────┘
```

## 📁 파일 구조

```
my_report/
├── streamlit_car_agent.py      # Streamlit 메인 앱
├── main.py                     # 공통 타입 정의
├── corrective_rag.py           # Corrective RAG Agent
├── car_agent.py               # 자동차 에이전트
├── enhanced_pdf_processor.py  # PDF 처리기
├── run_streamlit.py           # 실행 스크립트
├── requirements.txt           # 의존성 목록
└── README.md        # 이 파일
```

## 🔧 설정 옵션

### **Streamlit 설정**
- 포트: `--server.port 8501`
- 주소: `--server.address 0.0.0.0`
- 브라우저 자동 열기: `--browser.gatherUsageStats false`

### **에이전트 설정**
- LLM 모델: `gpt-4o-mini`
- 임베딩 모델: `text-embedding-3-small`
- 최대 반복 횟수: 3회

## 🐛 문제 해결

### **일반적인 문제들**

1. **OpenAI API 키 오류**
   - `.env` 파일에 올바른 API 키가 설정되었는지 확인
   - API 키에 충분한 크레딧이 있는지 확인

2. **벡터 DB 로드 실패**
   - `car_manual_chroma_db` 폴더가 존재하는지 확인
   - PDF 처리를 먼저 완료했는지 확인

3. **메모리 부족**
   - PDF 파일 크기를 줄이거나 청크 크기 조정
   - 배치 크기 감소

4. **Streamlit 실행 오류**
   - 포트가 이미 사용 중인 경우 다른 포트 사용
   - 의존성 패키지 재설치

### **로그 확인**
- 터미널에서 실행 로그 확인
- Streamlit 로그: `~/.streamlit/logs/`

## 📊 성능 최적화

### **PDF 처리 최적화**
- 청크 크기 조정 (기본: 800자)
- 배치 크기 조정 (기본: 50개)
- 임베딩 모델 변경 가능

### **응답 속도 개선**
- 캐싱 활용
- 불필요한 반복 제거
- 프롬프트 최적화

## 🤝 기여하기

1. 이슈 리포트
2. 기능 제안
3. 코드 개선
4. 문서 업데이트

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 지원

문제가 발생하거나 질문이 있으시면 이슈를 생성해주세요.

---

**즐거운 자동차 에이전트 사용을 위해! 🚗✨**
# llm_service_report

# 📄 PDF 논문 요약 Agent

PDF 논문을 자동으로 요약하고 질문에 답변하는 AI 기반 도구입니다. Streamlit 웹 인터페이스를 통해 쉽게 사용할 수 있습니다.

## 🚀 주요 기능

- **PDF 논문 자동 요약**: 업로드된 PDF 논문을 AI가 자동으로 요약
- **지능형 질의응답**: 논문 내용에 대한 질문에 정확한 답변 제공  
- **의미적 검색**: FAISS 벡터 스토어를 활용한 효율적인 문서 검색
- **대화형 인터페이스**: Streamlit 기반의 사용자 친화적 웹 UI
- **대화 히스토리 관리**: 이전 질문과 답변 기록 유지
- **다운로드 지원**: 생성된 요약을 TXT 파일로 다운로드

## 📋 필요 조건

- Python 3.7+
- OpenAI API 키

## 🛠️ 설치 방법

### 1. 저장소 클론
```bash
git clone https://github.com/[사용자명]/paper_summarizer.git
cd paper_summarizer
```

### 2. 가상환경 생성 및 활성화
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. 패키지 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

#### 선택적 설정 (기본값 사용 가능)
```env
OPENAI_MODEL=gpt-3.5-turbo
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SEARCH_K=3
HISTORY_LIMIT=3
SUMMARY_TYPE=comprehensive
```

## 📖 사용법

### 웹 인터페이스로 실행 (권장)
```bash
streamlit run paper_agent.py
```

브라우저에서 `http://localhost:8501`로 접속하여 사용하세요.

### 사용 단계
1. **PDF 업로드**: 사이드바에서 논문 PDF 파일 업로드
2. **요약 생성**: "📝 요약 생성" 버튼 클릭
3. **질문하기**: 요약 확인 후 논문에 대한 질문 입력
4. **답변 확인**: AI가 생성한 답변을 실시간으로 확인

## 📁 프로젝트 구조

```
PAPER_SUMMARIZER/
├── paper_agent.py      # 메인 애플리케이션
├── requirements.txt    # 필요 패키지 목록
├── .env               # 환경 변수 (생성 필요)
├── .gitignore         # Git 무시 파일
└── README.md          # 프로젝트 설명
```

## 📦 필요 패키지

```txt
langchain
langgraph  
langchain-openai
langchain-chroma
pypdf
faiss-cpu
streamlit
python-dotenv
```

## 🔧 설정 옵션

| 환경 변수 | 기본값 | 설명 |
|-----------|--------|------|
| `OPENAI_API_KEY` | (필수) | OpenAI API 키 |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | 사용할 OpenAI 모델 |
| `CHUNK_SIZE` | `1000` | 텍스트 청크 크기 |
| `CHUNK_OVERLAP` | `200` | 청크 간 중복 길이 |
| `SEARCH_K` | `3` | 검색 결과 개수 |
| `HISTORY_LIMIT` | `3` | 대화 히스토리 제한 |
| `SUMMARY_TYPE` | `comprehensive` | 요약 유형 (brief/structured/comprehensive) |

## 📊 지원 형식

- **입력**: PDF 파일
- **출력**: 웹 인터페이스 표시, TXT 파일 다운로드

## 💡 사용 팁

### 효과적인 질문 예시
- "이 논문의 주요 기여점은 무엇인가요?"
- "사용된 연구 방법론을 설명해주세요."
- "실험 결과의 핵심 내용을 요약해주세요."
- "이 연구의 한계점은 무엇인가요?"

### 요약 유형 설정
```env
SUMMARY_TYPE=brief          # 3-5줄 간단 요약
SUMMARY_TYPE=structured     # 구조화된 형태 요약
SUMMARY_TYPE=comprehensive  # 종합적인 상세 요약
```

## ✨ 기능 미리보기

### 요약 결과 예시
```
📄 논문 요약 결과

**연구 배경 및 동기**
이 연구는 자연어 처리 분야에서 트랜스포머 모델의 효율성을 개선하고자...

**주요 방법론**
- BERT 기반 아키텍처 개선
- 새로운 어텐션 메커니즘 도입
- 대규모 데이터셋을 활용한 사전 훈련

**핵심 결과**
기존 모델 대비 15% 성능 향상 및 30% 추론 속도 개선...
```

### 질의응답 예시
```
Q1: 이 논문에서 제안한 새로운 방법론의 핵심은 무엇인가요?
A1: 이 논문에서는 multi-head attention 메커니즘을 개선한 새로운 구조를 제안했습니다...

Q2: 실험 결과는 어떻게 평가되었나요?
A2: GLUE 벤치마크와 자체 데이터셋을 사용하여 성능을 평가했으며...
```

## 🐛 문제 해결

### 자주 발생하는 문제

**Q: "OPENAI_API_KEY가 설정되지 않았습니다" 오류**
```bash
# .env 파일이 프로젝트 루트에 있는지 확인
# API 키가 올바르게 설정되었는지 확인
```

**Q: PDF 업로드 후 처리 실패**
```bash
# PDF 파일이 텍스트 추출 가능한지 확인 (스캔본이 아닌 텍스트 PDF)
# 파일 크기가 너무 크지 않은지 확인 (권장: 10MB 이하)
```

**Q: 메모리 부족 오류**
```bash
# CHUNK_SIZE를 줄여보세요 (예: 500)
# 큰 PDF의 경우 부분적으로 처리
```

## 🔒 보안 및 개인정보

- API 키는 `.env` 파일에 저장하며 Git에 업로드되지 않음
- 업로드된 PDF는 임시로만 처리되며 서버에 저장되지 않음
- 대화 내용은 세션 동안만 메모리에 저장

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

## 🔗 관련 링크

- **저장소**: [https://github.com/toto6343/paper-summary-agent](https://github.com/toto6343/paper-summary-agent)
- **Issues**: [https://github.com/toto6343/paper-summary-agent/issues](https://github.com/toto6343/paper-summary-agent/issues)

## 🙏 사용된 기술

- **[LangChain](https://langchain.com/)** - LLM 애플리케이션 프레임워크
- **[OpenAI GPT](https://openai.com/)** - 자연어 처리 및 생성
- **[FAISS](https://faiss.ai/)** - 효율적인 유사도 검색
- **[Streamlit](https://streamlit.io/)** - 웹 애플리케이션 프레임워크
- **[PyPDF](https://pypdf.readthedocs.io/)** - PDF 텍스트 추출

## 📞 지원

문제가 발생하거나 제안사항이 있으시면 [GitHub Issues](https://github.com/toto6343/paper-summary-agent/issues)를 통해 알려주세요.

---

> 💡 **팁**: 처음 사용하시는 경우 작은 크기의 PDF로 테스트해보시기를 권장합니다.

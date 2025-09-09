# PDF 논문 요약 Agent - msgpack 직렬화 오류 수정 버전
# requirements.txt에 다음 패키지들을 추가해주세요:
# langchain
# langgraph  
# langchain-openai
# langchain-chroma
# pypdf
# faiss-cpu
# streamlit
# python-dotenv

import os
import tempfile
import pickle
from typing import List, Dict, Any, TypedDict, Annotated, Optional
from pathlib import Path

# dotenv 환경변수 로딩
from dotenv import load_dotenv
load_dotenv()  # .env 파일 로딩

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.tools import tool
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# =============================================================================
# 1단계: 상태 정의 (벡터 스토어 직렬화 문제 해결)
# =============================================================================

class AgentState(TypedDict):
    """Agent의 상태를 관리하는 클래스 - 직렬화 가능한 데이터만 포함"""
    messages: Annotated[list, add_messages]
    pdf_content: str
    chunks: List[str]  # 벡터 스토어 대신 청크를 저장
    summary: str
    qa_history: List[Dict]
    current_query: str
    vector_store_path: str  # 벡터 스토어 경로만 저장

# =============================================================================
# 2단계: 벡터 스토어 관리 클래스
# =============================================================================

class VectorStoreManager:
    """벡터 스토어를 관리하는 클래스"""
    
    def __init__(self):
        self.vector_stores = {}  # 메모리에 벡터 스토어 저장
        
    def create_vector_store(self, chunks: List[str], store_id: str = "default") -> str:
        """벡터 스토어 생성 및 ID 반환"""
        try:
            embeddings = OpenAIEmbeddings()
            documents = [Document(page_content=chunk) for chunk in chunks]
            
            # FAISS 사용 (메모리 내 저장이 더 안정적)
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=embeddings
            )
            
            # 메모리에 저장
            self.vector_stores[store_id] = vector_store
            return store_id
            
        except Exception as e:
            print(f"벡터 스토어 생성 오류: {str(e)}")
            return ""
    
    def get_vector_store(self, store_id: str):
        """벡터 스토어 조회"""
        return self.vector_stores.get(store_id)
    
    def search(self, store_id: str, query: str, k: int = 3) -> List[str]:
        """벡터 스토어에서 검색"""
        vector_store = self.get_vector_store(store_id)
        if vector_store:
            docs = vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        return []

# 전역 벡터 스토어 매니저
vector_manager = VectorStoreManager()

# =============================================================================
# 3단계: 단순화된 처리 함수들
# =============================================================================

def load_and_chunk_pdf(pdf_path: str) -> tuple[str, List[str]]:
    """PDF 로드 및 청킹"""
    try:
        # PDF 로드
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        content = "\n\n".join([page.page_content for page in pages])
        
        # 텍스트 청킹
        chunk_size = int(os.getenv("CHUNK_SIZE", 1000))
        overlap = int(os.getenv("CHUNK_OVERLAP", 200))
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " "]
        )
        chunks = text_splitter.split_text(content)
        
        return content, chunks
        
    except Exception as e:
        print(f"PDF 처리 오류: {str(e)}")
        return f"PDF 처리 오류: {str(e)}", []

def generate_summary(content: str, summary_type: str = "comprehensive") -> str:
    """논문 요약 생성"""
    try:
        if not content or len(content.strip()) < 10:
            return "요약할 내용이 없습니다."
        
        print(f"요약 생성 시작: {len(content)} 문자")
        
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        
        # API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            return "요약 생성 오류: OpenAI API 키가 설정되지 않았습니다."
        
        llm = ChatOpenAI(model=model_name, temperature=0)
        
        # 내용이 너무 길면 앞부분만 사용
        max_length = 8000 if summary_type == "comprehensive" else 5000
        content_for_summary = content[:max_length]
        if len(content) > max_length:
            content_for_summary += "\n\n[내용이 길어 일부만 요약에 사용됨]"
        
        if summary_type == "brief":
            prompt = f"""
다음 논문을 3-5줄로 간단히 요약해주세요:

{content_for_summary}

요약:
"""
        elif summary_type == "structured":
            prompt = f"""
다음 논문을 구조화된 형태로 요약해주세요:

{content_for_summary}

다음 형식으로 작성하세요:
1. 제목 및 저자
2. 연구 목적
3. 주요 방법론
4. 핵심 결과
5. 결론 및 의의
"""
        else:  # comprehensive
            prompt = f"""
다음 논문을 종합적으로 요약해주세요:

{content_for_summary}

포함할 내용:
- 연구 배경 및 동기
- 연구 문제 및 가설
- 방법론 및 실험 설계
- 주요 결과 및 발견사항
- 논의 및 제한점
- 결론 및 향후 연구 방향
"""
        
        print("OpenAI API 호출 시작")
        response = llm.invoke(prompt)
        print("요약 생성 완료")
        
        return response.content
        
    except Exception as e:
        error_msg = f"요약 생성 오류: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        return error_msg

def answer_question(question: str, vector_store_id: str, chat_history: List = None) -> str:
    """질문 답변 생성"""
    try:
        # 관련 문서 검색
        search_k = int(os.getenv("SEARCH_K", 3))
        context_docs = vector_manager.search(vector_store_id, question, k=search_k)
        context = "\n\n".join(context_docs)
        
        if not context:
            return "관련 정보를 찾을 수 없습니다."
        
        # LLM으로 답변 생성
        model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        llm = ChatOpenAI(model=model_name, temperature=0)
        
        history_context = ""
        if chat_history:
            history_limit = int(os.getenv("HISTORY_LIMIT", 3))
            recent_history = chat_history[-history_limit:]
            history_context = "\n이전 대화:\n" + "\n".join([
                f"Q: {item['question']}\nA: {item['answer']}" 
                for item in recent_history
            ])
        
        prompt = f"""
        다음 논문 내용을 바탕으로 질문에 답변해주세요:
        
        관련 내용:
        {context}
        
        {history_context}
        
        질문: {question}
        
        답변:
        """
        
        response = llm.invoke(prompt)
        return response.content
        
    except Exception as e:
        return f"질문 답변 오류: {str(e)}"

# =============================================================================
# 4단계: 단순화된 봇 클래스 (LangGraph 없이)
# =============================================================================

class PaperSummaryBot:
    def __init__(self):
        self.pdf_content = ""
        self.summary = ""
        self.chat_history = []
        self.vector_store_id = ""
        
    def process_pdf(self, pdf_file):
        """PDF 파일 처리"""
        if pdf_file is None:
            return "PDF 파일을 업로드해주세요."
        
        try:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                # Streamlit UploadedFile 객체 처리
                if hasattr(pdf_file, 'read'):
                    tmp_file.write(pdf_file.read())
                else:
                    tmp_file.write(pdf_file.getvalue())
                tmp_path = tmp_file.name
            
            # PDF 로드 및 청킹
            content, chunks = load_and_chunk_pdf(tmp_path)
            
            if not chunks:
                return "PDF 처리에 실패했습니다."
            
            # 벡터 스토어 생성
            import uuid
            store_id = str(uuid.uuid4())
            vector_store_id = vector_manager.create_vector_store(chunks, store_id)
            
            if not vector_store_id:
                return "벡터 스토어 생성에 실패했습니다."
            
            # 요약 생성
            summary_type = os.getenv("SUMMARY_TYPE", "comprehensive")
            summary = generate_summary(content, summary_type)
            
            # 상태 업데이트
            self.pdf_content = content
            self.summary = summary
            self.vector_store_id = vector_store_id
            self.chat_history = []
            
            # 임시 파일 정리
            os.unlink(tmp_path)
            
            return summary
            
        except Exception as e:
            return f"처리 중 오류가 발생했습니다: {str(e)}"
    
    def ask_question(self, question: str) -> str:
        """질문 답변"""
        if not question.strip():
            return "질문을 입력해주세요."
        
        if not self.vector_store_id:
            return "먼저 PDF를 업로드하고 처리해주세요."
        
        try:
            answer = answer_question(question, self.vector_store_id, self.chat_history)
            
            # 히스토리에 추가
            self.chat_history.append({
                "question": question,
                "answer": answer
            })
            
            return answer
            
        except Exception as e:
            return f"답변 생성 중 오류: {str(e)}"

# =============================================================================
# 환경변수 체크 함수
# =============================================================================

def check_environment():
    """환경변수 설정 확인"""
    if not os.getenv("OPENAI_API_KEY"):
        return False
    return True

# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    """Streamlit 메인 함수"""
    
    try:
        import streamlit as st
        
        st.set_page_config(
            page_title="📄 PDF 논문 요약 Agent",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # 타이틀
        st.title("📄 PDF 논문 요약 Agent")
        st.markdown("PDF 논문을 업로드하고 요약을 생성하거나 질문하세요!")
        
        # API 키 확인
        if not os.getenv("OPENAI_API_KEY"):
            st.error("🚨 OPENAI_API_KEY가 설정되지 않았습니다!")
            
            with st.expander("🔧 환경 설정 방법", expanded=True):
                st.info("프로젝트 루트에 .env 파일을 생성하고 API 키를 설정해주세요.")
                st.code("OPENAI_API_KEY=your-api-key-here")
                
                st.markdown("### 추가 설정 (선택사항)")
                st.code("""OPENAI_MODEL=gpt-3.5-turbo
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
SEARCH_K=3
HISTORY_LIMIT=3
SUMMARY_TYPE=comprehensive""")
            
            st.stop()
        
        # 세션 상태 초기화
        if 'bot' not in st.session_state:
            st.session_state.bot = PaperSummaryBot()
            
        if 'summary_displayed' not in st.session_state:
            st.session_state.summary_displayed = ""
        
        # 사이드바 - PDF 업로드
        with st.sidebar:
            st.header("📄 PDF 업로드")
            uploaded_file = st.file_uploader(
                "PDF 파일 선택",
                type=['pdf'],
                help="분석할 PDF 논문을 업로드하세요"
            )
            
            if uploaded_file and st.button("📝 요약 생성", type="primary"):
                with st.spinner("논문 분석 중..."):
                    try:
                        summary = st.session_state.bot.process_pdf(uploaded_file)
                        st.session_state.summary_displayed = summary
                        st.success("✅ 요약 생성 완료!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ 요약 생성 실패: {str(e)}")
            
            # 대화 초기화 버튼
            if st.session_state.bot.chat_history and st.button("🔄 대화 기록 초기화"):
                st.session_state.bot.chat_history = []
                st.success("대화 기록이 초기화되었습니다.")
                st.rerun()
        
        # 메인 영역
        if st.session_state.summary_displayed:
            st.markdown("### 📄 논문 요약 결과")
            st.markdown(st.session_state.summary_displayed)
            
            # 다운로드 버튼
            st.download_button(
                label="📥 요약 다운로드 (TXT)",
                data=st.session_state.summary_displayed,
                file_name="paper_summary.txt",
                mime="text/plain"
            )
            
            st.divider()
            
            # 질문 섹션
            st.markdown("### 💭 질문하기")
            
            # 채팅 히스토리 표시
            if st.session_state.bot.chat_history:
                st.markdown("#### 대화 기록")
                for i, qa in enumerate(st.session_state.bot.chat_history):
                    with st.container():
                        st.markdown(f"**Q{i+1}:** {qa['question']}")
                        st.markdown(f"**A{i+1}:** {qa['answer']}")
                        if i < len(st.session_state.bot.chat_history) - 1:
                            st.divider()
                
                st.divider()
            
            # 질문 입력
            with st.form("question_form", clear_on_submit=True):
                question = st.text_area(
                    "논문에 대해 질문해보세요:",
                    placeholder="예: 이 논문의 주요 기여점은 무엇인가요?",
                    height=100
                )
                submitted = st.form_submit_button("질문하기", type="primary")
                
                if submitted and question.strip():
                    with st.spinner("답변 생성 중..."):
                        try:
                            answer = st.session_state.bot.ask_question(question)
                            st.success("✅ 답변이 생성되었습니다!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"답변 생성 중 오류: {str(e)}")
        else:
            st.info("👈 사이드바에서 PDF를 업로드하고 요약 생성 버튼을 클릭하세요.")
            
            # 사용법 안내
            with st.expander("📋 사용법 안내", expanded=True):
                st.markdown("""
                ### 📋 사용 방법
                1. **PDF 업로드**: 사이드바에서 PDF 논문 파일을 업로드
                2. **요약 생성**: '요약 생성' 버튼을 클릭하여 논문 요약 생성
                3. **질문하기**: 요약 생성 후 논문에 대한 질문 입력
                4. **답변 확인**: AI가 생성한 답변을 확인
                
                ### 🔧 주요 기능
                - PDF 텍스트 추출 및 청킹
                - FAISS 벡터 스토어를 통한 의미적 검색
                - LLM 기반 논문 요약
                - 컨텍스트 기반 질문 답변
                - 대화 히스토리 관리
                
                ### ⚙️ 현재 설정
                - **모델**: {os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')}
                - **청크 크기**: {os.getenv('CHUNK_SIZE', '1000')}
                - **검색 결과 수**: {os.getenv('SEARCH_K', '3')}
                """)
    
    except ImportError as e:
        print(f"❌ 모듈 import 오류: {str(e)}")
        print("필요한 패키지를 설치해주세요: pip install streamlit")
    except Exception as e:
        print(f"❌ Streamlit 실행 오류: {str(e)}")

# =============================================================================
# 메인 실행부
# =============================================================================

if __name__ == "__main__":
    # 환경변수 체크
    if not check_environment():
        print("\n📁 프로젝트 루트에 .env 파일을 생성하고 다음 내용을 추가해주세요:")
        print("OPENAI_API_KEY=your-api-key-here")
        exit(1)
    
    # Streamlit으로 실행
    main()
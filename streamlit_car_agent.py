import streamlit as st
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import tempfile
from pathlib import Path

# 필요한 라이브러리 import
from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage

# 구현된 클래스들 import
from enhanced_pdf_processor import EnhancedPDFProcessor
from corrective_rag import LangGraphCorrectiveRAG
from car_agent import LangGraphCarAgent

# 페이지 설정
st.set_page_config(
    page_title="🚗 자동차 에이전트",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 스타일
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f8ff;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #ffe6e6;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e6ffe6;
        border-left: 4px solid #44ff44;
        margin: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #251327;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #5f5f5f;
        margin-right: 2rem;
    }
</style>
""",
    unsafe_allow_html=True,
)


def initialize_session_state():
    """세션 상태 초기화"""
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "car_agent" not in st.session_state:
        st.session_state.car_agent = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = "대기 중"


def load_existing_vector_db():
    """기존 벡터 DB 로드"""
    try:
        if os.path.exists("./car_manual_chroma_db"):
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vector_db = Chroma(
                persist_directory="./car_manual_chroma_db",
                embedding_function=embeddings,
                collection_name="car_manual",
            )

            # 문서 수 확인
            doc_count = vector_db._collection.count()

            st.session_state.vector_db = vector_db
            st.session_state.db_initialized = True
            st.session_state.processing_status = (
                f"기존 DB 로드 완료 ({doc_count}개 문서)"
            )

            return (
                True,
                f"기존 벡터 DB를 성공적으로 로드했습니다. (문서 수: {doc_count})",
            )
        else:
            return False, "기존 벡터 DB가 없습니다. 새로 생성해주세요."
    except Exception as e:
        return False, f"벡터 DB 로드 중 오류 발생: {str(e)}"


def process_pdf_file(uploaded_file):
    """PDF 파일 처리"""
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # PDF 처리기 초기화
        processor = EnhancedPDFProcessor(tmp_file_path)

        # 처리 진행 상황 표시
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 1단계: PDF 로드
        status_text.text("PDF 파일 로딩 중...")
        progress_bar.progress(20)
        processor.load_pdf()

        # 2단계: 구조 분석
        status_text.text("문서 구조 분석 중...")
        progress_bar.progress(40)
        structure_info = processor.analyze_document_structure()

        # 3단계: 청크 생성
        status_text.text("구조화된 청크 생성 중...")
        progress_bar.progress(60)
        chunks = processor.create_enhanced_chunks()

        # 4단계: 벡터 DB 생성
        status_text.text("벡터 데이터베이스 생성 중...")
        progress_bar.progress(80)
        vector_db = processor.create_vector_database(chunks)

        # 5단계: 완료
        status_text.text("처리 완료!")
        progress_bar.progress(100)

        # 세션 상태 업데이트
        st.session_state.vector_db = vector_db
        st.session_state.db_initialized = True
        st.session_state.processing_status = (
            f"PDF 처리 완료 ({structure_info['total_sections']}개 섹션)"
        )

        # 임시 파일 삭제
        os.unlink(tmp_file_path)

        return True, {
            "total_pages": structure_info["total_pages"],
            "total_sections": structure_info["total_sections"],
            "section_types": structure_info["section_types"],
            "tables": len(structure_info["tables"]),
            "warnings": len(structure_info["warnings"]),
            "chunks": len(chunks),
        }

    except Exception as e:
        return False, f"PDF 처리 중 오류 발생: {str(e)}"


def initialize_agents():
    """에이전트들 초기화"""
    try:
        if st.session_state.vector_db is None:
            return False, "벡터 DB가 초기화되지 않았습니다."

        # LLM 초기화
        if st.session_state.llm is None:
            st.session_state.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 자동차 에이전트 초기화
        if st.session_state.car_agent is None:
            st.session_state.car_agent = LangGraphCarAgent(
                st.session_state.vector_db, st.session_state.llm
            )

        return True, "에이전트 초기화 완료"

    except Exception as e:
        return False, f"에이전트 초기화 중 오류 발생: {str(e)}"


def display_chat_message(role: str, content: str, timestamp: str = None):
    """채팅 메시지 표시"""
    if role == "user":
        st.markdown(
            f"""
        <div class="chat-message user-message">
            <strong>👤 사용자</strong><br>
            {content}
            {f'<br><small style="color: #ff00ff;">{timestamp}</small>' if timestamp else ''}
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="chat-message assistant-message">
            <strong>🤖 자동차 에이전트</strong><br>
            {content}
            {f'<br><small style="color: #ff00ff;">{timestamp}</small>' if timestamp else ''}
        </div>
        """,
            unsafe_allow_html=True,
        )


def main():
    """메인 함수"""
    initialize_session_state()

    # 헤더
    st.markdown(
        '<div class="main-header">🚗 LangGraph 기반 자동차 에이전트</div>',
        unsafe_allow_html=True,
    )

    # 사이드바
    with st.sidebar:
        st.markdown("## 📊 시스템 상태")

        # 처리 상태
        st.markdown(f"**상태:** {st.session_state.processing_status}")

        # 벡터 DB 상태
        if st.session_state.db_initialized and st.session_state.vector_db:
            try:
                doc_count = st.session_state.vector_db._collection.count()
                st.markdown(f"**벡터 DB:** ✅ 활성 ({doc_count}개 문서)")
            except:
                st.markdown("**벡터 DB:** ❌ 오류")
        else:
            st.markdown("**벡터 DB:** ❌ 미초기화")

        # 에이전트 상태
        if st.session_state.car_agent:
            st.markdown("**자동차 에이전트:** ✅ 활성")
        else:
            st.markdown("**자동차 에이전트:** ❌ 미초기화")

        st.markdown("---")

        # 기존 DB 로드 버튼
        if st.button("🔄 기존 DB 로드", use_container_width=True):
            success, message = load_existing_vector_db()
            if success:
                st.success(message)
            else:
                st.error(message)

        # 채팅 기록 초기화
        if st.button("🗑️ 채팅 기록 초기화", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # 메인 컨텐츠
    tab1, tab2, tab3 = st.tabs(["💬 채팅", "📄 PDF 처리", "📊 시스템 정보"])

    with tab1:
        st.markdown(
            '<div class="section-header">💬 자동차 에이전트와 대화하기</div>',
            unsafe_allow_html=True,
        )

        # 에이전트 초기화 확인
        if not st.session_state.car_agent:
            if st.session_state.db_initialized:
                success, message = initialize_agents()
                if success:
                    st.success(message)
                else:
                    st.error(message)
            else:
                st.warning("먼저 PDF를 처리하거나 기존 벡터 DB를 로드해주세요.")

        # 채팅 인터페이스
        if st.session_state.car_agent:
            # 채팅 기록 표시
            for message in st.session_state.chat_history:
                display_chat_message(
                    message["role"], message["content"], message.get("timestamp")
                )

            # 사용자 입력
            user_input = st.text_input(
                "메시지를 입력하세요:",
                placeholder="예: 창문을 열어줘, 엔진 시동 방법을 알려줘, 안녕하세요",
                key="user_input",
            )

            col1, col2 = st.columns([1, 4])
            with col1:
                send_button = st.button("전송", use_container_width=True)

            # 메시지 전송 처리
            if send_button and user_input:
                # 사용자 메시지 추가
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_input, "timestamp": timestamp}
                )

                # 에이전트 응답 생성
                with st.spinner("응답을 생성하는 중..."):
                    try:
                        result = st.session_state.car_agent.process_input(user_input)

                        # 응답 메시지 추가
                        response_timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": result["final_response"],
                                "timestamp": response_timestamp,
                                "metadata": {
                                    "route_type": result["route_type"],
                                    "confidence": result["confidence"],
                                    "processing_time": result["processing_time"],
                                },
                            }
                        )

                        st.rerun()

                    except Exception as e:
                        error_timestamp = datetime.now().strftime("%H:%M:%S")
                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": f"죄송합니다. 오류가 발생했습니다: {str(e)}",
                                "timestamp": error_timestamp,
                            }
                        )
                        st.rerun()

    with tab2:
        st.markdown(
            '<div class="section-header">📄 PDF 파일 처리</div>', unsafe_allow_html=True
        )

        # PDF 업로드
        uploaded_file = st.file_uploader(
            "자동차 매뉴얼 PDF 파일을 업로드하세요:",
            type=["pdf"],
            help="PDF 파일을 업로드하면 자동으로 구조화되어 벡터 데이터베이스에 저장됩니다.",
        )

        if uploaded_file is not None:
            st.info(
                f"업로드된 파일: {uploaded_file.name} ({uploaded_file.size:,} bytes)"
            )

            if st.button("🚀 PDF 처리 시작", use_container_width=True):
                success, result = process_pdf_file(uploaded_file)

                if success:
                    st.success("PDF 처리가 완료되었습니다!")

                    # 처리 결과 표시
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("총 페이지 수", result["total_pages"])
                        st.metric("총 섹션 수", result["total_sections"])
                    with col2:
                        st.metric("표 개수", result["tables"])
                        st.metric("경고사항 개수", result["warnings"])
                    with col3:
                        st.metric("생성된 청크 수", result["chunks"])

                    # 섹션 타입별 분포
                    st.markdown("**섹션 타입별 분포:**")
                    for section_type, count in result["section_types"].items():
                        st.write(f"- {section_type}: {count}개")

                    # 에이전트 초기화
                    success, message = initialize_agents()
                    if success:
                        st.success("자동차 에이전트가 초기화되었습니다!")
                    else:
                        st.error(message)
                else:
                    st.error(result)

    with tab3:
        st.markdown(
            '<div class="section-header">📊 시스템 정보</div>', unsafe_allow_html=True
        )

        # 벡터 DB 정보
        if st.session_state.db_initialized and st.session_state.vector_db:
            try:
                doc_count = st.session_state.vector_db._collection.count()

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("벡터 DB 문서 수", doc_count)
                    st.metric("DB 상태", "활성")
                with col2:
                    st.metric("임베딩 모델", "text-embedding-3-small")
                    st.metric("컬렉션명", "car_manual")

                # DB 경로 정보
                st.markdown("**데이터베이스 경로:**")
                st.code("./car_manual_chroma_db", language="text")

            except Exception as e:
                st.error(f"벡터 DB 정보를 가져올 수 없습니다: {str(e)}")
        else:
            st.warning("벡터 DB가 초기화되지 않았습니다.")

        # 구조 분석 파일
        if os.path.exists("structure_analysis.json"):
            st.markdown("**구조 분석 파일:**")
            if st.button("📄 구조 분석 결과 보기"):
                with open("structure_analysis.json", "r", encoding="utf-8") as f:
                    analysis_data = json.load(f)

                st.json(analysis_data)

        # 채팅 기록 통계
        if st.session_state.chat_history:
            st.markdown("**채팅 기록 통계:**")
            user_messages = len(
                [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
            )
            assistant_messages = len(
                [
                    msg
                    for msg in st.session_state.chat_history
                    if msg["role"] == "assistant"
                ]
            )

            col1, col2 = st.columns(2)
            with col1:
                st.metric("사용자 메시지", user_messages)
            with col2:
                st.metric("에이전트 응답", assistant_messages)


if __name__ == "__main__":
    main()

"""
LangGraph 기반 자동차 에이전트 메인 모듈
"""

import os
import json
import re
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# Corrective RAG 상태 정의
class CorrectiveRAGState(TypedDict):
    """Corrective RAG Agent의 상태를 정의합니다."""

    # 입력
    question: str
    messages: List[BaseMessage]

    # 분석 결과
    analysis: Optional[Dict[str, Any]]
    keywords: List[str]
    complexity: str

    # 검색 및 컨텍스트
    search_results: List[Any]
    context: str
    refined_context: str

    # 답변 생성
    answer: str
    corrected_answer: str

    # 검증 및 품질
    is_correct: bool
    confidence: float
    issues: List[str]
    improvements: List[str]

    # 제어 변수
    iteration: int
    max_iterations: int
    is_satisfactory: bool
    should_continue: bool

    # 로그 및 디버깅
    correction_steps: List[Dict[str, Any]]
    error_message: Optional[str]


# 자동차 에이전트 상태 정의
class CarAgentState(TypedDict):
    """자동차 에이전트의 상태를 정의합니다."""

    # 입력
    user_input: str
    messages: List[BaseMessage]

    # 라우팅 정보
    route_type: Optional[Literal["car_control", "car_manual", "fallback"]]
    confidence: float

    # 자동차 제어 관련
    control_action: Optional[str]
    control_result: Optional[str]

    # 매뉴얼 검색 관련
    search_query: Optional[str]
    search_results: List[Any]
    manual_answer: Optional[str]

    # 일반 대화 관련
    general_response: Optional[str]

    # 최종 응답
    final_response: str

    # 메타데이터
    timestamp: str
    processing_time: float
    error_message: Optional[str]


@dataclass
class CorrectionStep:
    """수정 단계 정보를 저장하는 데이터 클래스"""

    step_number: int
    action: str
    description: str
    result: str
    confidence: float
    timestamp: str

"""
LangGraph 기반 자동차 에이전트
"""

import json
import re
from datetime import datetime
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from main import CarAgentState
from corrective_rag import LangGraphCorrectiveRAG


# LangGraph 노드들 구현
def route_input_node(state: CarAgentState) -> CarAgentState:
    """사용자 입력을 분석하여 적절한 노드로 라우팅합니다."""
    print(f"🔍 입력 분석 중: {state['user_input']}")

    try:
        # 라우팅 프롬프트
        routing_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 자동차 에이전트의 라우팅 시스템입니다. 
사용자의 입력을 분석하여 다음 중 하나로 분류해주세요:

1. **car_control**: 자동차 기능 제어 요청
   - 창문 열기/닫기, 조명 켜기/끄기, 오디오 조절, 에어컨/히터 조절 등
   - 키워드: "열어", "닫아", "켜", "꺼", "조절", "설정", "올려", "내려"

2. **car_manual**: 자동차 매뉴얼 검색 요청
   - 기능 설명, 사용법, 주의사항, 문제 해결 등
   - 키워드: "방법", "설명", "어떻게", "왜", "주의", "문제", "해결"

3. **fallback**: 일반적인 대화
   - 인사, 감사, 일상 대화, 자동차와 무관한 질문

분석 결과를 JSON 형식으로 제공해주세요:
{{
    "route_type": "car_control" | "car_manual" | "fallback",
    "confidence": 0.0-1.0,
    "reasoning": "분류 이유"
}}""",
                ),
                ("human", "{user_input}"),
            ]
        )

        # 라우팅 실행
        routing_chain = (
            routing_prompt
            | ChatOpenAI(model="gpt-4o-mini", temperature=0)
            | StrOutputParser()
        )
        routing_result = routing_chain.invoke({"user_input": state["user_input"]})

        # JSON 파싱
        routing_data = json.loads(routing_result)

        print(
            f"📊 라우팅 결과: {routing_data['route_type']} (신뢰도: {routing_data['confidence']:.2f})"
        )

        return {
            **state,
            "route_type": routing_data["route_type"],
            "confidence": routing_data["confidence"],
            "messages": state.get("messages", [])
            + [HumanMessage(content=f"라우팅: {routing_data['route_type']}")],
        }

    except Exception as e:
        print(f"❌ 라우팅 중 오류: {e}")
        return {
            **state,
            "route_type": "fallback",
            "confidence": 0.5,
            "error_message": str(e),
        }


def car_control_node(state: CarAgentState) -> CarAgentState:
    """자동차 제어 노드"""
    print("🚗 자동차 제어 처리 중...")

    try:
        # 자동차 제어 프롬프트
        control_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 자동차 제어 시스템입니다. 
사용자의 요청을 분석하여 구체적인 제어 명령을 생성해주세요.

지원하는 제어 기능:
1. 창문 제어: 열기/닫기, 올리기/내리기
2. 조명 제어: 전조등, 후미등, 실내등 켜기/끄기
3. 오디오 제어: 볼륨 조절, 음원 변경
4. 온도 제어: 에어컨/히터 온도 조절
5. 시트 제어: 시트 위치 조절

반드시 다음 JSON 형식으로만 응답해주세요. 다른 텍스트는 포함하지 마세요:
{{
    "action_type": "window",
    "action": "open",
    "target": "all",
    "value": null,
    "response": "창문을 열었습니다."
}}""",
                ),
                ("human", "{user_input}"),
            ]
        )

        # 제어 명령 생성
        control_chain = (
            control_prompt
            | ChatOpenAI(model="gpt-4o-mini", temperature=0)
            | StrOutputParser()
        )
        control_result = control_chain.invoke({"user_input": state["user_input"]})

        print(f"🔍 LLM 원본 응답: '{control_result}'")

        # 응답이 비어있는지 확인
        if not control_result or control_result.strip() == "":
            print("⚠️ LLM이 빈 응답을 반환했습니다.")
            control_data = {
                "action_type": "unknown",
                "action": "unknown",
                "target": "all",
                "value": None,
                "response": f"죄송합니다. '{state['user_input']}' 요청을 처리할 수 없습니다.",
            }
        else:
            # JSON 추출 시도
            control_data = None

            # 방법 1: 직접 JSON 파싱
            try:
                control_data = json.loads(control_result.strip())
                print("✅ 직접 JSON 파싱 성공")
            except json.JSONDecodeError:
                print("❌ 직접 JSON 파싱 실패, 정규식으로 JSON 추출 시도")

                # 방법 2: 정규식으로 JSON 추출
                json_match = re.search(r"\{.*\}", control_result, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group()
                        control_data = json.loads(json_str)
                        print("✅ 정규식 JSON 추출 성공")
                    except json.JSONDecodeError:
                        print("❌ 정규식 JSON 추출도 실패")

            # 모든 방법이 실패한 경우 기본값 사용
            if control_data is None:
                print("⚠️ 모든 JSON 파싱 방법 실패, 기본값 사용")
                control_data = {
                    "action_type": "unknown",
                    "action": "unknown",
                    "target": "all",
                    "value": None,
                    "response": f"죄송합니다. '{state['user_input']}' 요청을 처리할 수 없습니다.",
                }

        print(f"📊 최종 파싱된 데이터: {control_data}")

        # 필수 필드 확인 및 기본값 설정
        if "response" not in control_data or not control_data["response"]:
            control_data["response"] = f"'{state['user_input']}' 요청을 처리했습니다."
            print("⚠️ response 필드가 없어서 기본값으로 설정")

        # 시뮬레이션된 제어 실행
        control_action = f"{control_data.get('action_type', 'unknown')}_{control_data.get('action', 'unknown')}_{control_data.get('target', 'all')}"
        if control_data.get("value"):
            control_action += f"_{control_data['value']}"

        print(f"🎮 제어 명령 실행: {control_action}")
        print(f"💬 사용자 응답: {control_data['response']}")

        return {
            **state,
            "control_action": control_action,
            "control_result": control_data["response"],
            "final_response": control_data["response"],
            "messages": state.get("messages", [])
            + [AIMessage(content=f"자동차 제어: {control_data['response']}")],
        }

    except Exception as e:
        print(f"❌ 자동차 제어 중 오류: {e}")
        error_response = f"죄송합니다. 자동차 제어 중 오류가 발생했습니다: {str(e)}"
        return {
            **state,
            "control_action": "error",
            "control_result": error_response,
            "final_response": error_response,
            "error_message": str(e),
        }


def car_manual_node(state: CarAgentState) -> CarAgentState:
    """자동차 매뉴얼 검색 노드"""
    print("📖 자동차 매뉴얼 검색 처리 중...")

    try:
        # Corrective RAG Agent 생성 (임시)
        corrective_rag = LangGraphCorrectiveRAG(
            state.get("vector_db"),
            ChatOpenAI(model="gpt-4o-mini", temperature=0),
            max_iterations=3,
        )

        # Corrective RAG로 처리
        result = corrective_rag.process_question(state["user_input"])

        print(f"📝 Corrective RAG 답변 생성 완료")

        return {
            **state,
            "search_query": state["user_input"],
            "search_results": [],  # Corrective RAG 내부에서 처리됨
            "manual_answer": result["answer"],
            "final_response": result["answer"],
            "messages": state.get("messages", [])
            + [AIMessage(content=f"매뉴얼 검색: {result['answer'][:100]}...")],
        }

    except Exception as e:
        print(f"❌ 매뉴얼 검색 중 오류: {e}")
        return {
            **state,
            "search_query": state["user_input"],
            "search_results": [],
            "manual_answer": "매뉴얼 검색 중 오류가 발생했습니다.",
            "final_response": "죄송합니다. 매뉴얼 검색 중 오류가 발생했습니다.",
            "error_message": str(e),
        }


def fallback_node(state: CarAgentState) -> CarAgentState:
    """일반적인 대화를 처리하는 폴백 노드"""
    print("💬 일반 대화 처리 중...")

    try:
        # 일반 대화 프롬프트
        fallback_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 친근한 자동차 에이전트입니다. 
사용자와 자연스럽고 도움이 되는 대화를 나누어주세요.

대화 스타일:
1. 친근하고 정중한 톤
2. 자동차 관련 질문이면 도움을 제공
3. 일반적인 질문이면 적절히 응답
4. 필요시 자동차 기능을 안내
5. 사용자의 감정을 이해하고 공감

답변을 작성해주세요:""",
                ),
                ("human", "{user_input}"),
            ]
        )

        fallback_chain = (
            fallback_prompt
            | ChatOpenAI(model="gpt-4o-mini", temperature=0)
            | StrOutputParser()
        )
        general_response = fallback_chain.invoke({"user_input": state["user_input"]})

        print(f"💭 일반 대화 응답 생성 완료")

        return {
            **state,
            "general_response": general_response,
            "final_response": general_response,
            "messages": state.get("messages", [])
            + [AIMessage(content=f"일반 대화: {general_response}")],
        }

    except Exception as e:
        print(f"❌ 일반 대화 처리 중 오류: {e}")
        return {
            **state,
            "general_response": "죄송합니다. 응답을 생성할 수 없습니다.",
            "final_response": "죄송합니다. 응답을 생성할 수 없습니다.",
            "error_message": str(e),
        }


def route_to_node(
    state: CarAgentState,
) -> Literal["car_control", "car_manual", "fallback"]:
    """라우팅 결과에 따라 적절한 노드로 분기합니다."""
    route_type = state.get("route_type", "fallback")
    confidence = state.get("confidence", 0.0)

    print(f"🔄 라우팅: {route_type} (신뢰도: {confidence:.2f})")

    # 신뢰도가 낮으면 폴백으로 처리
    if confidence < 0.3:
        return "fallback"

    return route_type


class LangGraphCarAgent:
    """LangGraph 기반 자동차 에이전트"""

    def __init__(self, vector_db: Chroma, llm: ChatOpenAI):
        self.vector_db = vector_db
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """LangGraph 그래프를 구성합니다."""

        # 그래프 생성
        workflow = StateGraph(CarAgentState)

        # 노드 추가
        workflow.add_node("route_input", route_input_node)
        workflow.add_node("car_control", car_control_node)
        workflow.add_node("car_manual", car_manual_node)
        workflow.add_node("fallback", fallback_node)

        # 시작점 설정
        workflow.set_entry_point("route_input")

        # 조건부 라우팅
        workflow.add_conditional_edges(
            "route_input",
            route_to_node,
            {
                "car_control": "car_control",
                "car_manual": "car_manual",
                "fallback": "fallback",
            },
        )

        # 모든 노드에서 종료
        workflow.add_edge("car_control", END)
        workflow.add_edge("car_manual", END)
        workflow.add_edge("fallback", END)

        return workflow.compile()

    def process_input(self, user_input: str) -> Dict[str, Any]:
        """사용자 입력을 처리합니다."""
        print(f"🤖 LangGraph 자동차 에이전트가 입력을 처리합니다: '{user_input}'")

        start_time = datetime.now()

        # 초기 상태 설정
        initial_state = {
            "user_input": user_input,
            "messages": [HumanMessage(content=user_input)],
            "route_type": None,
            "confidence": 0.0,
            "control_action": None,
            "control_result": None,
            "search_query": None,
            "search_results": [],
            "manual_answer": None,
            "general_response": None,
            "final_response": "",
            "timestamp": datetime.now().isoformat(),
            "processing_time": 0.0,
            "error_message": None,
            "vector_db": self.vector_db,  # 벡터 DB 전달
        }

        # 그래프 실행
        try:
            final_state = self.graph.invoke(initial_state)

            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()

            # 결과 정리
            result = {
                "user_input": user_input,
                "route_type": final_state.get("route_type", "fallback"),
                "confidence": final_state.get("confidence", 0.0),
                "final_response": final_state.get("final_response", ""),
                "control_action": final_state.get("control_action"),
                "search_query": final_state.get("search_query"),
                "manual_answer": final_state.get("manual_answer"),
                "general_response": final_state.get("general_response"),
                "processing_time": processing_time,
                "timestamp": final_state.get("timestamp"),
                "error_message": final_state.get("error_message"),
            }

            print(
                f"\n🎉 처리 완료 - 라우팅: {result['route_type']}, 처리 시간: {processing_time:.2f}초"
            )

            return result

        except Exception as e:
            print(f"❌ 그래프 실행 중 오류: {e}")
            return {
                "user_input": user_input,
                "route_type": "error",
                "confidence": 0.0,
                "final_response": "처리 중 오류가 발생했습니다.",
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat(),
                "error_message": str(e),
            }

"""
LangGraph 기반 Corrective RAG Agent
"""

import json
from datetime import datetime
from typing import Dict, Any, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from main import CorrectiveRAGState, CorrectionStep


class LangGraphCorrectiveRAG:
    """LangGraph 기반 Corrective RAG Agent"""

    def __init__(self, vector_db: Chroma, llm: ChatOpenAI, max_iterations: int = 3):
        self.vector_db = vector_db
        self.llm = llm
        self.max_iterations = max_iterations

        # 프롬프트 템플릿들 설정
        self._setup_prompts()

        # 그래프 구성
        self.graph = self._build_graph()

    def _setup_prompts(self):
        """프롬프트 템플릿들을 설정합니다."""

        # 1. 질문 분석 프롬프트
        self.analysis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 자동차 매뉴얼 전문가입니다. 사용자의 질문을 분석하여 다음 정보를 제공해주세요:

1. 질문의 핵심 의도
2. 필요한 정보 유형 (기능 설명, 조작 방법, 주의사항 등)
3. 검색해야 할 키워드
4. 예상되는 답변 복잡도 (단순/중간/복잡)

분석 결과를 JSON 형식으로 제공해주세요:
{{
    "intent": "질문의 핵심 의도",
    "info_type": "필요한 정보 유형",
    "keywords": ["키워드1", "키워드2", "키워드3"],
    "complexity": "단순/중간/복잡",
    "expected_sections": ["예상 섹션1", "예상 섹션2"]
}}""",
                ),
                ("human", "{question}"),
            ]
        )

        # 2. 컨텍스트 개선 프롬프트
        self.context_refinement_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """다음은 자동차 매뉴얼에서 검색된 관련 정보입니다. 
사용자 질문에 답하기 위해 이 정보를 분석하고 개선해주세요.

다음 단계를 수행해주세요:
1. 컨텍스트의 관련성 평가
2. 부족한 정보 식별
3. 추가 검색이 필요한 키워드 제안
4. 개선된 컨텍스트 구성

개선된 컨텍스트를 제공해주세요.""",
                ),
                ("human", "질문: {question}\n컨텍스트: {context}"),
            ]
        )

        # 3. 답변 생성 프롬프트
        self.answer_generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 자동차 매뉴얼 전문가입니다. 
제공된 컨텍스트를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.

답변 작성 가이드라인:
1. 정확성: 매뉴얼의 정보를 정확히 반영
2. 명확성: 이해하기 쉽게 설명
3. 완전성: 질문에 대한 완전한 답변
4. 안전성: 안전 관련 정보는 강조
5. 구조화: 단계별로 명확하게 구성

답변을 작성해주세요:""",
                ),
                ("human", "질문: {question}\n컨텍스트: {context}"),
            ]
        )

        # 4. 답변 검증 프롬프트
        self.correction_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """당신은 답변 품질 검증 전문가입니다. 
제공된 답변을 검토하고 필요시 수정해주세요.

검토 기준:
1. 정확성: 정보가 올바른가?
2. 완전성: 질문에 완전히 답했는가?
3. 명확성: 이해하기 쉬운가?
4. 일관성: 논리적으로 일관된가?
5. 안전성: 안전 관련 정보가 적절히 포함되었는가?

검토 결과를 JSON 형식으로 제공해주세요:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "issues": ["문제점1", "문제점2"],
    "improvements": ["개선사항1", "개선사항2"],
    "corrected_answer": "수정된 답변"
}}""",
                ),
                ("human", "질문: {question}\n답변: {answer}\n컨텍스트: {context}"),
            ]
        )

    def _build_graph(self) -> StateGraph:
        """LangGraph 그래프를 구성합니다."""

        # 그래프 생성
        workflow = StateGraph(CorrectiveRAGState)

        # 노드 추가
        workflow.add_node("analyze_question", self._analyze_question_node)
        workflow.add_node("search_context", self._search_context_node)
        workflow.add_node("refine_context", self._refine_context_node)
        workflow.add_node("generate_answer", self._generate_answer_node)
        workflow.add_node("validate_answer", self._validate_answer_node)

        # 시작점 설정
        workflow.set_entry_point("analyze_question")

        # 엣지 추가
        workflow.add_edge("analyze_question", "search_context")

        # 조건부 라우팅: 컨텍스트 개선 여부
        workflow.add_conditional_edges(
            "search_context",
            self._should_refine_route,
            {"refine": "refine_context", "generate": "generate_answer"},
        )

        workflow.add_edge("refine_context", "generate_answer")
        workflow.add_edge("generate_answer", "validate_answer")

        # 조건부 라우팅: 계속 진행 여부
        workflow.add_conditional_edges(
            "validate_answer",
            self._should_continue_route,
            {"continue": "analyze_question", "end": END},  # 다시 분석부터 시작
        )

        return workflow.compile()

    def _analyze_question_node(self, state: CorrectiveRAGState) -> CorrectiveRAGState:
        """질문 분석 노드"""
        print(f"🔍 질문 분석 중: {state['question']}")

        try:
            # 질문 분석
            analysis_chain = (
                ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """당신은 자동차 매뉴얼 전문가입니다. 사용자의 질문을 분석하여 다음 정보를 제공해주세요:

1. 질문의 핵심 의도
2. 필요한 정보 유형 (기능 설명, 조작 방법, 주의사항 등)
3. 검색해야 할 키워드
4. 예상되는 답변 복잡도 (단순/중간/복잡)

분석 결과를 JSON 형식으로 제공해주세요:
{{
    "intent": "질문의 핵심 의도",
    "info_type": "필요한 정보 유형",
    "keywords": ["키워드1", "키워드2", "키워드3"],
    "complexity": "단순/중간/복잡",
    "expected_sections": ["예상 섹션1", "예상 섹션2"]
}}""",
                        ),
                        ("human", "{question}"),
                    ]
                )
                | self.llm
                | StrOutputParser()
            )
            analysis_result = analysis_chain.invoke({"question": state["question"]})

            # JSON 파싱
            analysis = json.loads(analysis_result)

            # 수정 단계 기록
            step = CorrectionStep(
                step_number=len(state.get("correction_steps", [])) + 1,
                action="analyze",
                description="질문 분석",
                result=analysis_result,
                confidence=0.9,
                timestamp=datetime.now().isoformat(),
            )

            return {
                **state,
                "analysis": analysis,
                "keywords": analysis.get("keywords", [state["question"]]),
                "complexity": analysis.get("complexity", "단순"),
                "correction_steps": state.get("correction_steps", []) + [step.__dict__],
                "messages": state.get("messages", [])
                + [HumanMessage(content=f"질문 분석: {state['question']}")],
            }

        except Exception as e:
            print(f"❌ 질문 분석 중 오류: {e}")
            return {
                **state,
                "analysis": {
                    "intent": "일반적인 질문",
                    "keywords": [state["question"]],
                    "complexity": "단순",
                },
                "keywords": [state["question"]],
                "complexity": "단순",
                "error_message": str(e),
            }

    def _search_context_node(self, state: CorrectiveRAGState) -> CorrectiveRAGState:
        """컨텍스트 검색 노드"""
        print(f"🔍 컨텍스트 검색 중...")

        try:
            # 키워드 기반 검색
            keywords = state.get("keywords", [state["question"]])
            search_query = " ".join(keywords)

            # 벡터 검색
            search_results = self.vector_db.similarity_search(search_query, k=5)

            # 컨텍스트 구성
            context_parts = []
            for i, result in enumerate(search_results):
                context_parts.append(f"[문서 {i+1}]\n{result.page_content}")

            context = "\n\n".join(context_parts)

            # 수정 단계 기록
            step = CorrectionStep(
                step_number=len(state.get("correction_steps", [])) + 1,
                action="search",
                description="컨텍스트 검색",
                result=f"검색된 문서: {len(search_results)}개",
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
            )

            return {
                **state,
                "search_results": search_results,
                "context": context,
                "correction_steps": state.get("correction_steps", []) + [step.__dict__],
                "messages": state.get("messages", [])
                + [
                    AIMessage(
                        content=f"컨텍스트 검색 완료: {len(search_results)}개 문서"
                    )
                ],
            }

        except Exception as e:
            print(f"❌ 컨텍스트 검색 중 오류: {e}")
            return {
                **state,
                "search_results": [],
                "context": "컨텍스트를 찾을 수 없습니다.",
                "error_message": str(e),
            }

    def _refine_context_node(self, state: CorrectiveRAGState) -> CorrectiveRAGState:
        """컨텍스트 개선 노드"""
        print(f"🔧 컨텍스트 개선 중...")

        try:
            # 컨텍스트 개선
            refinement_chain = (
                ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """다음은 자동차 매뉴얼에서 검색된 관련 정보입니다. 
사용자 질문에 답하기 위해 이 정보를 분석하고 개선해주세요.

다음 단계를 수행해주세요:
1. 컨텍스트의 관련성 평가
2. 부족한 정보 식별
3. 추가 검색이 필요한 키워드 제안
4. 개선된 컨텍스트 구성

개선된 컨텍스트를 제공해주세요.""",
                        ),
                        ("human", "질문: {question}\n컨텍스트: {context}"),
                    ]
                )
                | self.llm
                | StrOutputParser()
            )
            refined_context = refinement_chain.invoke(
                {"question": state["question"], "context": state["context"]}
            )

            # 수정 단계 기록
            step = CorrectionStep(
                step_number=len(state.get("correction_steps", [])) + 1,
                action="refine",
                description="컨텍스트 개선",
                result=f"개선된 컨텍스트 길이: {len(refined_context)}자",
                confidence=0.8,
                timestamp=datetime.now().isoformat(),
            )

            return {
                **state,
                "refined_context": refined_context,
                "correction_steps": state.get("correction_steps", []) + [step.__dict__],
                "messages": state.get("messages", [])
                + [AIMessage(content="컨텍스트 개선 완료")],
            }

        except Exception as e:
            print(f"❌ 컨텍스트 개선 중 오류: {e}")
            return {
                **state,
                "refined_context": state["context"],
                "error_message": str(e),
            }

    def _generate_answer_node(self, state: CorrectiveRAGState) -> CorrectiveRAGState:
        """답변 생성 노드"""
        print(f"💭 답변 생성 중...")

        try:
            # 답변 생성
            answer_chain = (
                ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """당신은 자동차 매뉴얼 전문가입니다. 
제공된 컨텍스트를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.

답변 작성 가이드라인:
1. 정확성: 매뉴얼의 정보를 정확히 반영
2. 명확성: 이해하기 쉽게 설명
3. 완전성: 질문에 대한 완전한 답변
4. 안전성: 안전 관련 정보는 강조
5. 구조화: 단계별로 명확하게 구성

답변을 작성해주세요:""",
                        ),
                        ("human", "질문: {question}\n컨텍스트: {context}"),
                    ]
                )
                | self.llm
                | StrOutputParser()
            )
            answer = answer_chain.invoke(
                {"question": state["question"], "context": state["refined_context"]}
            )

            # 수정 단계 기록
            step = CorrectionStep(
                step_number=len(state.get("correction_steps", [])) + 1,
                action="generate",
                description="답변 생성",
                result=f"답변 길이: {len(answer)}자",
                confidence=0.7,
                timestamp=datetime.now().isoformat(),
            )

            return {
                **state,
                "answer": answer,
                "correction_steps": state.get("correction_steps", []) + [step.__dict__],
                "messages": state.get("messages", [])
                + [AIMessage(content=f"답변 생성 완료: {answer[:100]}...")],
            }

        except Exception as e:
            print(f"❌ 답변 생성 중 오류: {e}")
            return {
                **state,
                "answer": "답변을 생성할 수 없습니다.",
                "error_message": str(e),
            }

    def _validate_answer_node(self, state: CorrectiveRAGState) -> CorrectiveRAGState:
        """답변 검증 노드"""
        print(f"✅ 답변 검증 중...")

        try:
            # 답변 검증
            correction_chain = (
                ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            """당신은 답변 품질 검증 전문가입니다. 
제공된 답변을 검토하고 필요시 수정해주세요.

검토 기준:
1. 정확성: 정보가 올바른가?
2. 완전성: 질문에 완전히 답했는가?
3. 명확성: 이해하기 쉬운가?
4. 일관성: 논리적으로 일관된가?
5. 안전성: 안전 관련 정보가 적절히 포함되었는가?

검토 결과를 JSON 형식으로 제공해주세요:
{{
    "is_correct": true/false,
    "confidence": 0.0-1.0,
    "issues": ["문제점1", "문제점2"],
    "improvements": ["개선사항1", "개선사항2"],
    "corrected_answer": "수정된 답변"
}}""",
                        ),
                        (
                            "human",
                            "질문: {question}\n답변: {answer}\n컨텍스트: {context}",
                        ),
                    ]
                )
                | self.llm
                | StrOutputParser()
            )
            correction_result = correction_chain.invoke(
                {
                    "question": state["question"],
                    "answer": state["answer"],
                    "context": state["refined_context"],
                }
            )

            # JSON 파싱
            correction_data = json.loads(correction_result)

            is_correct = correction_data.get("is_correct", True)
            confidence = correction_data.get("confidence", 0.5)
            corrected_answer = correction_data.get("corrected_answer", state["answer"])
            issues = correction_data.get("issues", [])
            improvements = correction_data.get("improvements", [])

            # 수정 단계 기록
            step = CorrectionStep(
                step_number=len(state.get("correction_steps", [])) + 1,
                action="validate",
                description="답변 검증",
                result=f"정확성: {is_correct}, 신뢰도: {confidence:.2f}",
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
            )

            return {
                **state,
                "corrected_answer": corrected_answer,
                "is_correct": is_correct,
                "confidence": confidence,
                "issues": issues,
                "improvements": improvements,
                "correction_steps": state.get("correction_steps", []) + [step.__dict__],
                "messages": state.get("messages", [])
                + [AIMessage(content=f"답변 검증 완료 - 신뢰도: {confidence:.2f}")],
            }

        except Exception as e:
            print(f"❌ 답변 검증 중 오류: {e}")
            return {
                **state,
                "corrected_answer": state["answer"],
                "is_correct": True,
                "confidence": 0.5,
                "issues": [],
                "improvements": [],
                "error_message": str(e),
            }

    def _should_refine_route(
        self, state: CorrectiveRAGState
    ) -> Literal["refine", "generate"]:
        """컨텍스트 개선이 필요한지 결정하는 라우팅 함수"""
        iteration = state.get("iteration", 0)
        complexity = state.get("complexity", "단순")

        # 첫 번째 반복이거나 복잡한 질문인 경우 컨텍스트 개선
        if iteration == 0 or complexity in ["중간", "복잡"]:
            return "refine"
        else:
            return "generate"

    def _should_continue_route(
        self, state: CorrectiveRAGState
    ) -> Literal["continue", "end"]:
        """계속 진행할지 결정하는 라우팅 함수"""
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)
        is_correct = state.get("is_correct", True)
        confidence = state.get("confidence", 0.0)

        # 최대 반복 횟수에 도달했거나 만족스러운 답변을 얻었으면 종료
        if iteration >= max_iterations or (is_correct and confidence > 0.7):
            return "end"
        else:
            return "continue"

    def process_question(self, question: str) -> Dict[str, Any]:
        """질문을 처리하는 메인 메서드"""
        print(f"🤖 LangGraph Corrective RAG Agent가 질문을 처리합니다: '{question}'")

        # 초기 상태 설정
        initial_state = {
            "question": question,
            "messages": [HumanMessage(content=question)],
            "analysis": None,
            "keywords": [],
            "complexity": "단순",
            "search_results": [],
            "context": "",
            "refined_context": "",
            "answer": "",
            "corrected_answer": "",
            "is_correct": False,
            "confidence": 0.0,
            "issues": [],
            "improvements": [],
            "iteration": 0,
            "max_iterations": self.max_iterations,
            "is_satisfactory": False,
            "should_continue": True,
            "correction_steps": [],
            "error_message": None,
        }

        # 그래프 실행
        try:
            final_state = self.graph.invoke(initial_state)

            # 결과 정리
            result = {
                "question": question,
                "answer": final_state.get(
                    "corrected_answer", final_state.get("answer", "")
                ),
                "is_satisfactory": final_state.get("is_correct", False)
                and final_state.get("confidence", 0) > 0.7,
                "confidence": final_state.get("confidence", 0.0),
                "iterations": final_state.get("iteration", 0),
                "correction_steps": final_state.get("correction_steps", []),
                "issues": final_state.get("issues", []),
                "improvements": final_state.get("improvements", []),
                "messages": final_state.get("messages", []),
                "timestamp": datetime.now().isoformat(),
            }

            print(
                f"\n🎉 처리 완료 - 신뢰도: {result['confidence']:.2f}, 반복 횟수: {result['iterations']}"
            )

            return result

        except Exception as e:
            print(f"❌ 그래프 실행 중 오류: {e}")
            return {
                "question": question,
                "answer": "처리 중 오류가 발생했습니다.",
                "is_satisfactory": False,
                "confidence": 0.0,
                "iterations": 0,
                "correction_steps": [],
                "issues": [str(e)],
                "improvements": [],
                "messages": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

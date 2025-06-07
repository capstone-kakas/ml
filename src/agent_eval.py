import json
from src.state.conditionRag import *
from src.state.reviewRag import *
from src.state.verifyRag import *
from typing import Annotated, List, Optional, TypedDict, Literal
from operator import add
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from textwrap import dedent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent
from IPython.display import Image, display
from langchain_core.messages import HumanMessage
from src.node import *

# 질문 라우팅 노드 
def make_agent_eval():
    
    nodes = {
        "analyze_question": analyze_question_tool_search,
        "verify_claim": verify_rag_node,
        "reviews_history": review_rag_node,
        "condition_guideline": condition_rag_node,
        "generate_answer": answer_final,
        "llm_fallback": llm_fallback,
        "evaluate_answer": evaluate_answer_node, 
    }

    # 그래프 생성을 위한 StateGraph 객체를 정의
    search_builder = StateGraph(ResearchAgentState)

    # 노드 추가
    for node_name, node_func in nodes.items():
        search_builder.add_node(node_name, node_func)

    # 엣지 추가 (병렬 처리)
    search_builder.add_edge(START, "analyze_question")
    search_builder.add_conditional_edges(
        "analyze_question",
        route_datasources_tool_search,
        ["verify_claim","reviews_history", "condition_guideline", "llm_fallback"]
    )

    # 검색 노드들을 generate_answer에 연결
    for node in ["verify_claim","reviews_history", "condition_guideline"]:
        search_builder.add_edge(node, "generate_answer")

    search_builder.add_edge("generate_answer", "evaluate_answer")

    # HITL 결과에 따른 조건부 엣지 추가
    search_builder.add_edge(
        "evaluate_answer", END
        # human_review,
        # {
        #     "approved": END,
        #     "rejected": "analyze_question"  # 승인되지 않은 경우 질문 분석 단계로 돌아감
        # }
    )

    search_builder.add_edge("llm_fallback", END)

    # 그래프 컴파일
    legal_rag_agent = search_builder.compile()

    # 그래프 시각화 
    return legal_rag_agent
    
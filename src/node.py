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


# 메인 그래프 상태 정의
class ResearchAgentState(TypedDict):
    question: str
    answers: Annotated[List[str], add]
    final_answer: str
    datasources: List[str]
    evaluation_report: Optional[dict]
    user_decision: Optional[str]



# 라우팅 결정을 위한 데이터 모델
class ToolSelector(BaseModel):
    """Routes the user question to the most appropriate tool."""
    tool: Literal["verify_claim", "reviews_history", "condition_guideline"] = Field(
        description="Select one of the tools, based on the user's question.",
    )

class ToolSelectors(BaseModel):
    """Select the appropriate tools that are suitable for the user question."""
    tools: List[ToolSelector] = Field(
        description="Select one or more tools, based on the user's question.",
    )

def analyze_question_tool_search(state: ResearchAgentState):
        question = state["question"]
        result = question_tool_router.invoke({"question": question})
        datasources = [tool.tool for tool in result.tools]
        return {"datasources": datasources}


def route_datasources_tool_search(state: ResearchAgentState) -> List[str]:
    datasources = set(state['datasources'])
    valid_sources = {"verify_claim", "reviews_history", "condition_guideline"}
    
    if datasources.issubset(valid_sources):
        return list(datasources)
    
    return list(valid_sources)
verify_web_agent, review_web_agent, condition_web_agent = make_verify_workflow(), make_review_workflow(), make_condition_workflow()
# 노드 정의 
def verify_rag_node(state: VerifyRagState, input=ResearchAgentState) -> ResearchAgentState:
    print("--- 진위여부 판별 전문가 에이전트 시작 ---")
    question = state["question"]
    answer = verify_web_agent.invoke({"question": question})
    return {"answers": [answer["node_answer"]]}

def review_rag_node(state: ReviewRagState, input=ResearchAgentState) -> ResearchAgentState:
    print("--- 리뷰 및 리콜 이력 검색 전문가 에이전트 시작 ---")
    question = state["question"]
    answer = review_web_agent.invoke({"question": question})
    return {"answers": [answer["node_answer"]]}

def condition_rag_node(state: ConditionRagState, input=ResearchAgentState) -> ResearchAgentState:
    print("--- 제품 상태 판별 전문가 에이전트 시작 ---")
    question = state["question"]
    answer = condition_web_agent.invoke({"question": question})
    return {"answers": [answer["node_answer"]]}


# 구조화된 출력을 위한 LLM 설정
structured_llm_tool_selector = llm.with_structured_output(ToolSelectors)

# 라우팅을 위한 프롬프트 템플릿
system = dedent("""You are an AI assistant specializing in routing user questions to the appropriate tools.
Use the following guidelines:
- For questions specifically about if the given information is verified and real, use the verify_claim tool.
- For questions specifically about customer review and experience and recall history, use the reviews_history tool.
- For questions specifically about how the condition of the product affects the price, use the condition_guideline tool.
Always choose all of the appropriate tools based on the user's question.""")

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

# 질문 라우터 정의
question_tool_router = route_prompt | structured_llm_tool_selector

# 최종 답변 생성 노드


# RAG 프롬프트 정의
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant answering questions based on provided documents. Follow these guidelines:

1. Use only information from the given documents.
2. If the document lacks relevant info, say "제공된 정보로는 충분한 답변을 할 수 없습니다."
3. Cite the source of information for each sentence in your answer. Use the following format:
    - For web sources: "출처 제목 (URL)"
4. Don't speculate or add information not in the documents.
5. Keep answers concise and clear.
6. Omit irrelevant information.
7. If multiple sources provide the same information, cite all relevant sources.
8. If information comes from multiple sources, combine them coherently while citing each source.

Example of citation usage:
"일부 사용자들은 디지털 에디션의 저장 용량이 제한적이라는 점을 지적하며, 추가 저장 장치 구매의 필요성을 언급하고 있습니다 (출처: IGN (www.ign.com/articles/playstation-5-review))"
"""
    ),
    ("human", "Answer the following question using these documents:\n\n[Documents]\n{documents}\n\n[Question]\n{question}"),
])

def answer_final(state: ResearchAgentState) -> ResearchAgentState:
    """
    Generate answer using the retrieved_documents
    """
    print("---최종 답변---")
    question = state["question"]
    documents = state.get("answers", [])
    if not isinstance(documents, list):
        documents = [documents]

    # 문서 내용을 문자열로 결합 
    documents_text = "\n\n".join(documents)

    # RAG generation
    rag_chain = rag_prompt | llm | StrOutputParser()
    generation = rag_chain.invoke({"documents": documents_text, "question": question})
    return {"final_answer": generation, "question":question}


# LLM Fallback 프롬프트 정의
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant helping with various topics. Follow these guidelines:

1. Provide accurate and helpful information to the best of your ability.
2. Express uncertainty when unsure; avoid speculation.
3. Keep answers concise yet informative.
4. Respond ethically and constructively.
5. Mention reliable general sources when applicable."""),
    ("human", "{question}"),
])

def llm_fallback(state: ResearchAgentState) -> ResearchAgentState:
    """
    Generate answer using the LLM without context
    """
    print("---Fallback 답변---")
    question = state["question"]
    
    # LLM chain
    llm_chain = fallback_prompt | llm | StrOutputParser()
    
    generation = llm_chain.invoke({"question": question})
    return {"final_answer": generation, "question":question}
evaluation_prompt = dedent("""
당신은 AI 어시스턴트가 생성한 답변을 평가하는 전문가입니다. 주어진 질문과 답변을 평가하고, 60점 만점으로 점수를 매기세요. 다음 기준을 사용하여 평가하십시오:

1. 정확성 (10점)
2. 관련성 (10점)
3. 완전성 (10점)
4. 인용 정확성 (10점)
5. 명확성과 간결성 (10점)
6. 객관성 (10점)

평가 과정:
1. 주어진 질문과 답변을 주의 깊게 읽으십시오.
2. 필요한 경우, 다음 도구를 사용하여 추가 정보를 수집하세요:
- verify_claim_with_web: 진위 여부 판단
- get_product_reviews_and_history: 제품 리뷰 및 리콜 이력 검색
- get_condition_guidelines: 제품 상태가 가격에 미치는 영향 검색색

도구 사용 형식:
Action: [tool_name]
Action Input: [input for the tool]

3. 각 기준에 대해 1-10점 사이의 점수를 매기세요.
4. 총점을 계산하세요 (60점 만점).

출력 형식:
{
"scores": {
    "accuracy": 0,
    "relevance": 0,
    "completeness": 0,
    "citation_accuracy": 0,
    "clarity_conciseness": 0,
    "objectivity": 0
},
"total_score": 0,
"brief_evaluation": "간단한 평가 설명"
}

최종 출력에는 각 기준의 점수, 총점, 그리고 간단한 평가 설명만 포함하세요.
""")


# 그래프 생성 
answer_reviewer = create_react_agent(
    llm, 
    tools=tools, 
    state_modifier=evaluation_prompt,
    )

# 답변 평가하는 노드를 추가
def evaluate_answer_node(state:ResearchAgentState):
    question = state["question"]
    final_answer = state["final_answer"]

    messages = [HumanMessage(content=f"""[질문]\n\{question}n\n[답변]\n{final_answer}""")]
    response = answer_reviewer.invoke({"messages": messages})
    response_dict = json.loads(response['messages'][-1].content)

    return {"evaluation_report": response_dict, "question": question, "final_answer": final_answer}
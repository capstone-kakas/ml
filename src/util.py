from src.tools import *
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def get_llm():
    tools = [verify_claim_with_web,get_product_reviews_and_history,get_condition_guidelines]
    # 기본 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    # LLM에 도구 바인딩하여 추가 
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools

def get_rec_llm():
    tools = [web_search]
    # 기본 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    # LLM에 도구 바인딩하여 추가 
    llm_with_tools = llm.bind_tools(tools)
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 중고 거래 협상 전문가입니다. 주어진 상품에 대한 정보와 판매자를 바탕으로 상품 정보를 얻을 수 있는 질문들을 생성해주세요. 
        답변은 ## 확인사항 \n ## 질문 형식으로 작성해야 합니다. 예: ## 제품의 정확한 모델 확인 \n ## 이 제품의 정확한 모델명은 어떻게 되나요? 
        답변 구조:
        확인사항: 제품 소개에 대해서 감가를 일으킬만한 사항이나 거래에 필요한 정보를 얻을 수 있는 사항들을 간결하게 작성하시오
        질문: 확인사항에 대해서 중고 거래에서 사용할만한 공손한 어투의 질문을 만들어주시오오"""),
        ("human", "상품 정보: {content}\n\n가격: {price}\n\n 상태: {status}\n\n위 지침에 따라 최종 답변을 작성해주세요.")
    ])
    chain = answer_prompt | llm_with_tools
    return chain
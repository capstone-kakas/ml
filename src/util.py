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

def get_rec_llm(chat=False):
    tools = [web_search]
    # 기본 LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
    # LLM에 도구 바인딩하여 추가 
    llm_with_tools = llm.bind_tools(tools)
    if not chat:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 중고 거래 협상 전문가입니다. 중고 거래 판매자와 구마재의 채팅을 바탕으로 진위 여부가 의심되는 부분이나 추가적인 정보를 얻을 수 있는 질문들을 생성해주세요. 
            형식적인 질문들이 아니라, 해당 게임기 제품에 대한 전문가 수준의 지식을 갖춘체 물어봐야 합니다
            답변은 ## 확인사항 \n ## 질문 형식으로 작성해야 합니다. 예시: ## 제품의 정확한 모델 확인 \n ## 이 제품의 정확한 모델명은 어떻게 되나요? 
            답변 구조:
            확인사항: 제품 소개에 대해서 감가를 일으킬만한 사항이나 거래에 필요한 정보를 얻을 수 있는 사항들을 간결하게 작성하시오
            질문: 확인사항에 대해서 중고 거래에서 사용할만한 공손한 어투의 질문을 4개부터 6개 사이로 만들어주시오"""),
            ("human", "상품 정보: {content}\n\n가격: {price}\n\n 상태: {status}\n\n위 지침에 따라 최종 답변을 작성해주세요.")
        ])
    else:
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 중고 거래 협상 전문가입니다. 중고 거래 판매자와 구마재의 채팅을 바탕으로 진위 여부가 의심되는 부분이나 추가적인 정보를 얻을 수 있는 질문들을 생성해주세요. 
            형식적인 질문들이 아니라, 해당 게임기 제품에 대한 전문가 수준의 지식을 갖춘체 물어봐야 합니다
            답변은 ## 확인사항 \n ## 질문 형식으로 작성해야 합니다. 예: ## 상품 용량 확인 \n ## 해당 제품의 용량이 어떻게 되나요? 
            답변 구조:
            확인사항: 제품 소개에 대해서 감가를 일으킬만한 사항이나 거래에 필요한 정보를 얻을 수 있는 사항들을 간결하게 작성하시오
            질문: 확인사항에 대해서 중고 거래에서 사용할만한 공손한 어투의 질문을 4개부터 6개 사이로 만들어주시오"""),
            ("human", "제품명: {productName} 판매자 채팅: {chat}\n\n위 지침에 따라 최종 답변을 작성해주세요.")
        ])
    chain = answer_prompt | llm_with_tools
    return chain
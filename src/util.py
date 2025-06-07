from src.tools import *
from langchain_openai import ChatOpenAI


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

    return llm_with_tools
from langchain_chroma import Chroma
from langchain_ollama  import OllamaEmbeddings
from langchain_core.documents import Document

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.tools import tool
from typing import List

# 문서 임베딩 모델
embeddings_model = OllamaEmbeddings(model="bge-m3") 

# Re-rank 모델
rerank_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
cross_reranker = CrossEncoderReranker(model=rerank_model, top_n=2)

# 웹 검색
web_retriever = ContextualCompressionRetriever(
    base_compressor=cross_reranker, 
    base_retriever=TavilySearchAPIRetriever(k=10),
)

@tool
def web_search(query: str) -> List[str]:
    """데이터베이스에 없는 정보 또는 최신 정보를 웹에서 검색합니다."""

    docs = web_retriever.invoke(query)

    formatted_docs = []
    for doc in docs:
        formatted_docs.append(
            Document(
                page_content= f'<Document href="{doc.metadata["source"]}"/>\n{doc.page_content}\n</Document>',
                metadata={"source": "web search", "url": doc.metadata["source"]}
            )
        )

    if len(formatted_docs) > 0:
        return formatted_docs
    
    return [Document(page_content="관련 정보를 찾을 수 없습니다.")]

@tool
def verify_claim_with_web(query: str) -> List[str]:
    """
    판매자의 주장 또는 정보의 진위를 검증하기 위한 웹 검색을 수행합니다.
    예: '이 제품은 2023년형이고 단종되었다고 합니다' 등
    """

    docs = web_retriever.invoke(query)

    formatted_docs = []
    for doc in docs:
        formatted_docs.append(
            Document(
                page_content=f'[출처]({doc.metadata["source"]})\n\n{doc.page_content}',
                metadata={"type": "claim_verification", "url": doc.metadata["source"]}
            )
        )

    return formatted_docs if formatted_docs else [Document(page_content="해당 주장에 대한 명확한 정보를 찾지 못했습니다.")]


@tool
def get_product_reviews_and_history(query: str) -> List[str]:
    """
    제품 이름을 바탕으로 리뷰, 사용자 경험, 리콜 이력 등을 검색합니다.
    예: 'LG 그램 2022 16인치' 등
    """

    docs = web_retriever.invoke(query)

    formatted_docs = []
    for doc in docs:
        formatted_docs.append(
            Document(
                page_content=f'[출처]({doc.metadata["source"]})\n\n{doc.page_content}',
                metadata={"type": "product_review", "url": doc.metadata["source"]}
            )
        )

    return formatted_docs if formatted_docs else [Document(page_content="해당 제품에 대한 리뷰 정보를 찾지 못했습니다.")]

@tool
def get_condition_guidelines(query: str) -> List[str]:
    """
    특정 제품군의 상태가 거래에 어떤 영향을 미치는지 기준 정보나 사례를 검색합니다.
    예: '아이폰 13 사용감 상태 거래 기준' 등
    """

    docs = web_retriever.invoke(query)

    formatted_docs = []
    for doc in docs:
        formatted_docs.append(
            Document(
                page_content=f'[출처]({doc.metadata["source"]})\n\n{doc.page_content}',
                metadata={"type": "condition_reference", "url": doc.metadata["source"]}
            )
        )

    return formatted_docs if formatted_docs else [Document(page_content="상태 기준 정보를 찾지 못했습니다.")]

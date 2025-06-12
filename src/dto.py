from fastapi import FastAPI, Form, Request
import uvicorn
# from src.api.index import api_router
from src.agent import *
from src.agent_eval import *

class ChatRequest(BaseModel):
    question: str

class ChatSellerRequest(BaseModel):
    question: str
    seller_chat : str
    chatTitle: str
    chatContent: str
    price: str
    status: str
    productName: str


class RecommendRequest(BaseModel):
    chatTitle: str
    chatContent: str
    price: str
    status: str
    chat: List[str]

class RecommendChatRequest(BaseModel):
    chat: List[str]
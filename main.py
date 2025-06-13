from fastapi import FastAPI, Form, Request
import uvicorn
# from src.api.index import api_router
from src.agent import *
from src.agent_eval import *
import time
from src.dto import *
from src.util import *

app = FastAPI(docs_url="/api-docs", openapi_url="/open-api-docs")


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # try:
        kakas_agent = make_agent()  # agent 객체 생성
        inputs = {"question": request.question}
        
        # LangGraph agent의 stream 응답 수집
        response = ''
        start = time.time()
        for output in kakas_agent.stream(inputs):
            for key, value in output.items():
                for k, v in value.items():
                    if k == 'final_answer':
                            response = v
        end = time.time()
        latency = end - start
        return {"response": response, 'latency' : latency}

    # except Exception as e:
    #     return {"error": str(e)}
    

@app.post("/chat-seller")
async def chat_seller_endpoint(request: ChatSellerRequest):
    try:
        question = '판매자 채팅: ' + request.seller_chat + '질문: ' + request.question + '\n 제품: ' + request.productName + '에 대해서 판매자 채팅 중에 거짓이 없는지 판별해주시오'
        kakas_agent = make_agent()  # agent 객체 생성
        inputs = {"question": question}
        
        # LangGraph agent의 stream 응답 수집
        response = ''
        start = time.time()
        for output in kakas_agent.stream(inputs):
            for key, value in output.items():
                for k, v in value.items():
                    if k == 'final_answer':
                            response = v
        end = time.time()
        latency = end - start
        return {"response": response, 'latency' : latency}

    except Exception as e:
        return {"error": str(e)}

@app.post("/chat-eval")
async def chat_eval_endpoint(request: ChatRequest):
    try:
        kakas_agent = make_agent_eval()  # agent 객체 생성
        inputs = {"question": request.question}
        
        # LangGraph agent의 stream 응답 수집
        response = ''
        eval = {}
        start = time.time()
        for output in kakas_agent.stream(inputs):
            for key, value in output.items():
                for k, v in value.items():
                    if k == 'final_answer':
                            response = v
                    if k == 'evaluation_report':
                            eval = v
        end = time.time()
        latency = end - start
        return {"response": response, "evaluation_report":eval, 'latency' : latency}
    except Exception as e:
        return {"error": str(e)}
    

@app.post("/recommend")
async def recommend_endpoint(request: RecommendRequest):
    try:
        kakas_agent = get_rec_llm()  # agent 객체 생성
        start = time.time()
        query = {"content" : request.chatTitle+"\n"+request.chatContent, "price" : request.price, "status" : request.status}
        response = kakas_agent.invoke(query).content
        end = time.time()
        latency = end - start
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        response_list = []

        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                heading = lines[i].replace("##", "").strip()
                question = lines[i + 1].replace("##", "").strip()
                response_list.append(heading+':'+question)
        return {"response": response_list, 'latency' : latency}
        
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/recommend-chat")
async def recommend_chat_endpoint(request: RecommendChatRequest):
    try:
        chat='\n'.join(l for l in request.chat)
        kakas_agent = get_rec_llm(chat=True)  # agent 객체 생성
        start = time.time()
        query = {"chat" : chat}
        response = kakas_agent.invoke(query).content
        end = time.time()
        latency = end - start
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        response_list = []

        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                heading = lines[i].replace("##", "").strip()
                question = lines[i + 1].replace("##", "").strip()
                response_list.append(heading+':'+question)
        return {"response": response_list, 'latency' : latency}
        
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
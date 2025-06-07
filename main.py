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
    try:
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
        chat='\n'.join(l[0]+': '+l[1] for l in request.chat)
        question = '상품 설명: ' + request.chatTitle + '('+request.status+') - ' + request.price + '\n' + request.chatContent + "\n의 상태를 더 자세히 확인하기 위한 질문을 만들고 싶다. 채팅 내역: " + chat + "또한 고려해서 질문들을 생성해라 \n 답변은  ## 확인사항 , ## 질문 형식으로 중고 거래에서 사용할 공손한 말투로 생성하시오(예시: ## 게임기 모델 확인 \n ## 플레이스테이션 5 모델이 어떻게 되나요?) 숫자나 기호를 매기지 말고 확인사항과 질문만 출력해라 "
        kakas_agent = get_rec_llm()  # agent 객체 생성
        start = time.time()
        response = kakas_agent.invoke(question).content
        end = time.time()
        latency = end - start
        lines = [line.strip() for line in response.strip().splitlines() if line.strip()]
        response_dict = {}

        for i in range(0, len(lines), 2):
            if i + 1 < len(lines):
                heading = lines[i].replace("##", "").strip()
                question = lines[i + 1].replace("##", "").strip()
                response_dict[heading] = question
        return {"response": response_dict, 'latency' : latency}
        
    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
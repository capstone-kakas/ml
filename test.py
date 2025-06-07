from src.agent import *

if __name__=='__main__':
    kakas_agent = make_agent()  # agent 객체 생성
    inputs = {"question": 'ps5 디지털 에디션 500GB의 최신 가격 동향이 어떻게 되나요?'}
    response = ''
    eval = {}
    for output in kakas_agent.stream(inputs):
        for key, value in output.items():
            for k, v in value.items():
                print(k, v)
                
    print(response)
    print(eval)


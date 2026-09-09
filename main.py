from pydantic import BaseModel
from langchain.agents import create_agent, AgentState
from dotenv import load_dotenv

load_dotenv()

class Answer(BaseModel):
    summary: str
    confidence: float


# define a custom state

class MyState(AgentState):
    user_id: str
    call_count: int

agent = create_agent(model="google_genai:gemini-3.6-flash", tools= [], response_format=Answer, state_schema=MyState)
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Summarize AI trends"
            }
        ],
        "user_id": "user_123",
        "call_count": 0
    })

print(result["structured_response"])
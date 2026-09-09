from langchain.agents import create_agent
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class Answer(BaseModel):
    summary: str
    confidence: float


# checkpointer stores the agent state , after execution 
agent = create_agent(
    model="google_genai:gemini-3.6-flash", tools=[], checkpointer=InMemorySaver(), response_format=Answer
)

# config is additional configuration for how the agent should run.
#here we configured a unique id for the conversation thread/state

config = {"configurable": {"thread_id": str(uuid7())}}

#pass the unique id , so that this state can be saved in chekpointer with this id 

result = agent.invoke(input=
    {"messages": [
        {
            "role":"user",
            "content": "What is the weather of San Francisco"
        }
    ]},
    config=config,
)

print(result["structured_response"])

#if we again invoke the agent with different input , but pass the same config , it searches and retrives the state corresponding to this id
#this way we gave it memory across differnt invocations

result = agent.invoke(
    input={
        "messages": [
            {
                "role": "user",
                "content": "What about romorrow"
            }
        ]
    },
    config=config
)

print(result["structured_response"])

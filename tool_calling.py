from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

# create an instance of the model
model = init_chat_model("google_genai:gemini-3.6-flash")


@tool
def get_weather(location: str) -> str:
    """Get the weather at a location."""
    return f"It's sunny in {location}."


# to make the tools avaiable to the model , we must bind them using bind_tools.
model_with_tools = model.bind_tools([get_weather])

messages = [
    {
        "role": "user",
        "content": "What's the weather in Boston?"
    }
]


response = model_with_tools.invoke(messages)
messages.append(response)

for tool_call in response.tool_calls:
    # print(f"Tool: {tool_call['name']}")
    # print(f"Args: {tool_call['args']}")
    
    tool_result = get_weather.invoke(tool_call)
    messages.append(tool_result)  #modify the state

final_response = model_with_tools.invoke(messages)
print(final_response.content[0]["text"])
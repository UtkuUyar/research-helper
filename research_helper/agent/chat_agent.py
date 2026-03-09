from langchain.agents import create_agent

from .tools import RetrieveContextTool
from research_helper.utils import get_chat_agent_prompt


def build_chat_agent(llm, paper_handler):
    retrieve_context = RetrieveContextTool(vs_handler=paper_handler.vec_db_handler)

    tools = [
        retrieve_context
    ]

    prompt = get_chat_agent_prompt()    
    agent = create_agent(llm, tools, system_prompt=prompt)

    return agent
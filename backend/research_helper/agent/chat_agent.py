from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver  

from .tools import RetrieveContextTool
from research_helper.utils import get_chat_agent_prompt


def build_chat_agent(paper_handler, paper_name=None):
    retrieve_context = RetrieveContextTool(vs_handler=paper_handler.vec_db_handler)

    tools = [
        retrieve_context
    ]

    paper_summary = getattr(paper_handler, "paper_summary", None)
    paper_name = getattr(paper_handler, "title", None)
    
    prompt = get_chat_agent_prompt(paper_name=paper_name, paper_summary=paper_summary)
    agent = create_agent(
        paper_handler.llm, 
        tools, 
        system_prompt=prompt, 
        checkpointer=InMemorySaver()
    )

    return agent
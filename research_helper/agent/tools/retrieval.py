from pydantic import BaseModel, Field
from typing import Type
from langchain.tools import BaseTool

from research_helper.handlers import VectorStoreHandler


class RetrieveInput(BaseModel):
    query: str = Field(description="Search query for retrieving paper context")


class RetrieveContextTool(BaseTool):

    name: str = "retrieve_context"
    description: str = "Retrieve relevant parts of the paper to answer a question."
    
    args_schema: Type[BaseModel] = RetrieveInput
    
    vs_handler: VectorStoreHandler

    def _run(self, query: str):

        docs = self.vs_handler.similarity_search(query, k=8)

        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in docs
        )

        return serialized
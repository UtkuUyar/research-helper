from pathlib import Path
from langchain_ollama import ChatOllama

from research_helper.handlers import PaperHandler
from research_helper.agent import build_chat_agent

def chat_with_paper(agent):

    print("\nPaper Assistant Ready.")
    print("Ask questions about the paper.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        try:
            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                }
            )

            answer = response["messages"][-1].content
            print(f"\nAssistant: {answer}\n")

        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    paper_name = "wanderlust"
    file_path = f".\data\{paper_name}.pdf"
    llm = ChatOllama(model="llama3.1:8b")
    
    output_dir = Path(f".\output\{paper_name}")
    output_dir.mkdir(exist_ok=True, parents=True)

    paper_handler = PaperHandler(llm=llm, file_path=file_path, output_dir=output_dir, summarize=False)
    agent = build_chat_agent(llm, paper_handler)

    chat_with_paper(agent)

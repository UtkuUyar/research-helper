from pathlib import Path
from dotenv import load_dotenv

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

            # for event in agent.stream(
            #     {"messages": [{"role": "user", "content": question}]},
            #     stream_mode="values",
            # ):
            #     event["messages"][-1].pretty_print()

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ]
                }, {"configurable": {"thread_id": "1"}}
            )

            answer = response["messages"][-1].content
            print(f"\nAssistant: {answer}\n")

        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    load_dotenv()

    paper_name = "wanderlust"
    file_path = f".\data\{paper_name}.pdf"
    
    output_dir = Path(f".\output\{paper_name}")
    output_dir.mkdir(exist_ok=True, parents=True)

    paper_handler = PaperHandler(file_path=file_path, output_dir=output_dir, summarize=True)
    agent = build_chat_agent(paper_handler)

    chat_with_paper(agent)

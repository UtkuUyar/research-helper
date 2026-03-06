from pathlib import Path
from langchain_ollama import ChatOllama

from research_helper.handlers import PaperHandler

if __name__ == "__main__":
    paper_name = "sample_paper_fasterrcnn"
    file_path = f".\data\{paper_name}.pdf"
    llm = ChatOllama(model="llama3:8b")
    
    output_dir = Path(f".\output\{paper_name}")
    output_dir.mkdir(exist_ok=True, parents=True)

    handler = PaperHandler(llm=llm, file_path=file_path, output_dir=output_dir)
    handler.save()
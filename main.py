import re
import json
import fitz
import pymupdf4llm

from pydantic import BaseModel
from typing import List

# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage


class SectionSummary(BaseModel):
    summary: str
    key_points: List[str]


def pdf_to_markdown(file_path):
    return pymupdf4llm.to_markdown(file_path)

def split_sections(md_text):
    pattern = r"^\s*\*\*(.+?)\*\*\s*$"

    parts = re.split(pattern, md_text, flags=re.MULTILINE)

    sections = []
    for i in range(1, len(parts), 2):
        title = parts[i].replace("**", "").strip()
        content = parts[i+1].strip() if i+1 < len(parts) else ""

        sections.append({
            "title": title,
            "content": content
        })

    return sections

def section_summary(section, llm):
    system_prompt = (
        "You are an expert research assistant specialized in analyzing scientific papers.\n\n"
        "Your task is to read a section of a research paper and produce an accurate and concise summary.\n\n"
        "Rules:\n"
        "- Only use information that appears in the provided text.\n"
        "- Do NOT hallucinate or add external knowledge.\n"
        "- Preserve technical terminology from the paper.\n"
        "- Focus on the main ideas, methods, findings, and conclusions in the section.\n"
        "- Ignore figure numbers, table references, page numbers, and formatting artifacts.\n\n"
        "Your output must contain two parts:\n\n"
        "1. Section Summary\n"
        "A concise explanation (3–5 sentences) describing the purpose and main ideas of the section.\n\n"
        "2. Key Points\n"
        "A bullet list of the most important technical details, such as:\n"
        "- problem definitions\n"
        "- methods or algorithms\n"
        "- datasets\n"
        "- evaluation metrics\n"
        "- experimental findings\n"
        "- important assumptions\n"
    )

    section_prompt = (
        "Section Title:\n"
        "{title}\n\n"
        "Section Content:\n"
        "{text}\n\n"
        "Please summarize this section."
    )

    # agent = create_agent(
    #     model=llm.with_structured_output(SectionSummary),
    #     system_prompt=SystemMessage(system_prompt)
    # )

    # result = agent.invoke(
    #     {
    #         "messages": [
    #             HumanMessage(
    #                 section_prompt.format(
    #                     title=section["title"],
    #                     text=section["content"]
    #                 )
    #             )
    #         ]
    #     }
    # )

    structured_llm = llm.with_structured_output(SectionSummary)
    sm = structured_llm.invoke([
        system_prompt, 
        section_prompt.format(
            title=section["title"],
            text=section["content"]
        )]
    )

    return sm

if __name__ == "__main__":
    file_path = ".\data\wanderlust.pdf"
    md_text = pdf_to_markdown(file_path)
    
    sections = split_sections(md_text)

    llm = ChatOllama(model="llama3:8b")

    secsum = section_summary(sections[4], llm)
    
    with open("secsum.json", "w") as f:
        json.dump(secsum.model_dump(), f, indent=2)

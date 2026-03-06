import re
import json
import fitz
import pymupdf4llm
from pathlib import Path
from tqdm import tqdm

from pydantic import BaseModel
from typing import List

from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.messages import SystemMessage, HumanMessage


class SectionSummary(BaseModel):
    summary: str
    key_points: List[str]
    important_entities: List[str]

class PaperSummary(BaseModel):
    research_problem: str
    key_contributions: List[str]
    method_overview: str
    experimental_findings: List[str]
    limitations: List[str]


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
        "- Do not omit important concrete details when they appear in the text.\n"
        "- Prefer extracting specific information instead of producing generic descriptions."
        " Examples of concrete details include:\n"
        "   + names of methods, models, or algorithms\n"
        "   + datasets, corpora, or experimental environments\n"
        "   + evaluation metrics, benchmarks, or performance measures\n"
        "   + numerical results, improvements, or comparisons\n"
        "   + experimental setups, procedures, or protocols\n"
        "   + theoretical assumptions, definitions, or formal results\n\n"
        "Your output must contain three parts:\n\n"
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
        "3. Important Entities\n"
        "List important technical entities mentioned in the section, such as:\n"
        "- models\n"
        "- datasets\n"
        "- algorithms\n"
        "- evaluation metrics\n"
        "- benchmarks\n"
    )

    section_prompt = (
        "Section Title:\n"
        "{title}\n\n"
        "Section Content:\n"
        "{text}\n\n"
        "Please summarize this section."
    )

    structured_llm = llm.with_structured_output(SectionSummary)
    sm = structured_llm.invoke([
        system_prompt, 
        section_prompt.format(
            title=section["title"],
            text=section["content"]
        )]
    )

    return sm

def paper_summary(section_summaries, llm):

    structured_llm = llm.with_structured_output(PaperSummary)

    system_prompt = (
        "You are an expert research assistant analyzing a scientific paper.\n\n"
        "You will receive summaries of different sections of a paper.\n"
        "Using only this information, generate a structured summary of the paper.\n\n"
        "Each field in the output has a specific meaning:\n\n"
        "research_problem:\n"
        "Explain the core problem or research question the paper is trying to solve.\n"
        "Describe the motivation and why the problem matters.\n\n"
        "key_contributions:\n"
        "List the main contributions claimed by the paper. These should be the novel\n"
        "ideas, datasets, algorithms, or frameworks introduced by the authors.\n\n"
        "method_overview:\n"
        "Explain the main approach or methodology proposed in the paper. Summarize\n"
        "how the system or algorithm works at a high level.\n\n"
        "experimental_findings:\n"
        "Summarize the most important experimental results or empirical findings.\n"
        "Focus on improvements, benchmarks, or evaluation outcomes.\n\n"
        "limitations:\n"
        "List known weaknesses, assumptions, or limitations of the approach.\n"
        "Only include limitations explicitly mentioned in the text. Otherwise return an empty list.\n\n"
        "Rules:\n"
        "- Only use information from the provided section summaries.\n"
        "- Do not hallucinate missing details.\n"
        "- Keep explanations concise but informative.\n"
    )

    prompt = (
        "Below are summaries of the paper's sections.\n\n"
        "{sections}\n\n"
        "Using these summaries, produce the structured paper summary."
    )

    sections_list = []
    for title, s in section_summaries.items():
        key_points = "\n".join(f"\t-{p}" for p in s.key_points)

        section_text = (
            f"Section Title: {title}\n\n"
            f"Summary:\n{s.summary}\n"
            f"Key Points:\n{key_points}"
        )
        sections_list.append(section_text)

    sections_text = "\n\n".join(sections_list)

    result = structured_llm.invoke(
        system_prompt + prompt.format(sections=sections_text)
    )

    return result

if __name__ == "__main__":
    paper_name = "sample_paper_fasterrcnn"
    file_path = f".\data\{paper_name}.pdf"
    llm = ChatOllama(model="llama3:8b")
    
    output_dir = Path(f".\output\{paper_name}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("Parsing the pdf as markdown text...")
    md_text = pdf_to_markdown(file_path)
    (output_dir / "md_text.md").write_bytes(md_text.encode())
    
    print("Splitting the paper into sections...")
    sections = split_sections(md_text)
    with open(output_dir / "sections.json", "w") as f:
        json.dump(sections, f, indent=2)

    print(f"Total of {len(sections)} sections found. Generating summaries for each section...")
    sec_summaries = {s["title"]:section_summary(s, llm) for s in tqdm(sections)}
    with open(output_dir / "section_summaries.json", "w") as f:
        json.dump({k: v.model_dump() for k, v in sec_summaries.items()}, f, indent=2)

    print("Generating the paper summary...")
    paper_sum = paper_summary(sec_summaries, llm)
    
    print("Summary done. Saving the results...")
    with open(output_dir / "paper_summary.json", "w") as f:
        json.dump(paper_sum.model_dump(), f, indent=2)

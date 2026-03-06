import json
import re
import pymupdf4llm

from pydantic import BaseModel
from typing import List
from langchain.messages import SystemMessage, HumanMessage

from research_helper.utils import get_section_summary_prompts, get_paper_summary_prompts
from .rag import ChunkHandler, VectorStoreHandler

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


class PaperHandler:
    def __init__(
            self,
            llm,
            chunk_size=800,
            chunk_overlap=150,
            embedding_model="nomic-embed-text",
            summarize=True,
            file_path=None,
            output_dir=None
        ):

        self.llm = llm
        self.output_dir = output_dir
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = ChunkHandler(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        self.embedding_model = embedding_model
        self.vec_db_handler = VectorStoreHandler(
            embedding_model=embedding_model
        )
        
        self.summarize = summarize
        if file_path is not None:
            self.process_paper(file_path)

    def process_paper(self, file_path):
        self.file_path = file_path
        self.md_text = self.read_document()
        self.sections = self.split_sections()
        
        if self.summarize:
            self.section_summaries = self.summarize_sections()
            self.paper_summary = self.summarize_paper()

        self.chunks = self.chunker.chunk_sections(self.sections)
        self.vec_db_handler.build_index(self.chunks)

    def clear(self):
        self.file_path = None
        self.md_text = None
        self.sections = None
        self.section_summaries = None
        self.paper_summary = None

        self.chunks = None
        self.vec_db_handler.clear()

    def save(self):
        (self.output_dir / "md_text.md").write_bytes(self.md_text.encode())
        
        with open(self.output_dir / "sections.json", "w") as f:
            json.dump(self.sections, f, indent=2)

        with open(self.output_dir / "section_summaries.json", "w") as f:
            json.dump({k: v.model_dump() for k, v in self.section_summaries.items()}, f, indent=2)

        with open(self.output_dir / "paper_summary.json", "w") as f:
            json.dump(self.paper_summary.model_dump(), f, indent=2)

    def read_document(self):
        return pymupdf4llm.to_markdown(self.file_path)
    
    def split_sections(self):
        assert self.md_text != None
        
        pattern = r"^\s*\*\*(.+?)\*\*\s*$"

        parts = re.split(pattern, self.md_text, flags=re.MULTILINE)

        sections = []
        for i in range(1, len(parts), 2):
            title = parts[i].replace("**", "").strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""

            sections.append({
                "title": title,
                "content": content
            })

        return sections
    
    def _section_summary(self, section):
        system_prompt, section_prompt = get_section_summary_prompts(section)
        structured_llm = self.llm.with_structured_output(SectionSummary)
        
        return structured_llm.invoke(
            [
                SystemMessage(system_prompt), 
                HumanMessage(section_prompt)
            ]
        )

    def summarize_sections(self):
        assert self.sections != None

        return {s["title"]:self._section_summary(s) for s in self.sections}

    def summarize_paper(self):
        assert self.section_summaries != None

        system_prompt, paper_prompt = get_paper_summary_prompts(self.section_summaries)
        structured_llm = self.llm.with_structured_output(PaperSummary)

        return structured_llm.invoke(
            [
                SystemMessage(system_prompt), 
                HumanMessage(paper_prompt)
            ]
        )
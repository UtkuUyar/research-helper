from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class ChunkHandler:
    def __init__(self, chunk_size=800, chunk_overlap=150):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk_sections(self, sections):

        documents = []

        for section in sections:

            title = section["title"]
            text = section["content"]

            chunks = self.splitter.split_text(text)

            for chunk in chunks:

                documents.append(
                    Document(
                        page_content=f"Section: {title}\n\n{chunk}",
                        metadata={
                            "section": title
                        }
                    )
                )

        return documents
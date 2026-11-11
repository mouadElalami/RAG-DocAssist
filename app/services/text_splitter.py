from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import settings

class TextSplitterService:
    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits list of raw documents into smaller semantic chunks,
        attaching chunk indices and unique chunk IDs to metadata.
        """
        chunks = self.splitter.split_documents(documents)
        
        for idx, chunk in enumerate(chunks):
            doc_id = chunk.metadata.get("doc_id", "doc")
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = f"{doc_id}_c{idx}"
            chunk.metadata["chunk_char_count"] = len(chunk.page_content)
        
        return chunks

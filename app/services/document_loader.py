import os
import uuid
from pathlib import Path
from typing import List, Tuple
from datetime import datetime
from langchain_core.documents import Document

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None


class DocumentLoaderService:
    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".docx", ".csv", ".json"}

    @classmethod
    def load_file(cls, file_path: Path, original_filename: str) -> Tuple[List[Document], str]:
        """
        Loads a file from path and returns list of Langchain Document objects along with a unique doc_id.
        """
        ext = file_path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Extension non supportée: {ext}. Formats supportés: {', '.join(cls.SUPPORTED_EXTENSIONS)}")

        doc_id = str(uuid.uuid4())[:8]
        docs: List[Document] = []
        file_size = file_path.stat().st_size
        indexed_at = datetime.utcnow().isoformat()

        base_metadata = {
            "doc_id": doc_id,
            "filename": original_filename,
            "file_type": ext.lstrip("."),
            "file_size": file_size,
            "indexed_at": indexed_at,
            "source": original_filename,
        }

        if ext == ".pdf":
            docs = cls._load_pdf(file_path, base_metadata)
        elif ext == ".docx":
            docs = cls._load_docx(file_path, base_metadata)
        elif ext in {".txt", ".md", ".markdown", ".csv", ".json"}:
            docs = cls._load_text(file_path, base_metadata)

        if not docs or not any(d.page_content.strip() for d in docs):
            raise ValueError(f"Le fichier '{original_filename}' est vide ou aucun texte extractible n'a été trouvé.")

        return docs, doc_id

    @classmethod
    def _load_pdf(cls, file_path: Path, base_metadata: dict) -> List[Document]:
        if not pypdf:
            raise ImportError("pypdf n'est pas installé.")
        
        docs = []
        with open(file_path, "rb") as f:
            reader = pypdf.PdfReader(f)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    meta = base_metadata.copy()
                    meta["page"] = page_num
                    docs.append(Document(page_content=text, metadata=meta))
        return docs

    @classmethod
    def _load_docx(cls, file_path: Path, base_metadata: dict) -> List[Document]:
        if not docx:
            raise ImportError("python-docx n'est pas installé.")
        
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text)
        
        combined_text = "\n\n".join(full_text)
        meta = base_metadata.copy()
        meta["page"] = 1
        return [Document(page_content=combined_text, metadata=meta)]

    @classmethod
    def _load_text(cls, file_path: Path, base_metadata: dict) -> List[Document]:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        
        meta = base_metadata.copy()
        meta["page"] = 1
        return [Document(page_content=text, metadata=meta)]

    @classmethod
    def load_raw_text(cls, title: str, text: str) -> Tuple[List[Document], str]:
        """
        Creates LangChain Document from direct raw text.
        """
        doc_id = str(uuid.uuid4())[:8]
        indexed_at = datetime.utcnow().isoformat()
        metadata = {
            "doc_id": doc_id,
            "filename": title,
            "file_type": "text_note",
            "file_size": len(text.encode("utf-8")),
            "indexed_at": indexed_at,
            "source": title,
            "page": 1
        }
        return [Document(page_content=text, metadata=metadata)], doc_id

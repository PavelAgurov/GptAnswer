"""
    Backend
"""

# pylint: disable=C0301,C0103,C0303,C0411,C0304

import os
from enum import Enum
import logging
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from langchain.docstore.document import Document
from langchain.vectorstores import Qdrant
from langchain.embeddings.base import Embeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings

from qdrant_client import QdrantClient

logger : logging.Logger = logging.getLogger()

@dataclass
class Answer:
    """Class for LLM answer"""
    answer      : str
    tokens_used : int

@dataclass_json
@dataclass
class SearchResult:
    """Result of the search"""
    content   : str
    score     : float
    metadata  : {}

class EmbeddingType(Enum):
    """Types of embeddings"""
    SBERT    = "SBERT (https://www.sbert.net/)"
    MULTILP  = "paraphrase-multilingual-MiniLM-L12-v2" 
    OPENAI35 = "Open AI Embeddings"

class LlmEmbeddingError(Exception):
    """Lmm embedding exception"""

class Backend():
    """Backend class"""
    
    score_threshold = 0.4
    sample_count = 3

    embeddings : Embeddings

    __CHUNKS_COLLECTION_NAME = 'chunks'
    __DISK_FOLDER = '.qindex'
    __INDEX_FOLDER = 'index'
    __QUERY_EXAMPLES  = 'query_examples.txt'

    def __init__(self):
        self.embeddings = self.get_embeddings(EmbeddingType.SBERT.name)

    def __get_api_key(self):
        return os.environ["OPENAI_API_KEY"]

    def get_embeddings(self, embedding_name : EmbeddingType)-> (OpenAIEmbeddings | SentenceTransformerEmbeddings):
        """Embeddings"""
        
        if embedding_name == EmbeddingType.OPENAI35.name:
            # https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html
            return OpenAIEmbeddings(openai_api_key= self.__get_api_key())
        
        if embedding_name == EmbeddingType.SBERT.name:
            # https://www.sbert.net/
            return SentenceTransformerEmbeddings(
                model_name= 'sentence-transformers/all-MiniLM-L6-v2',
                model_kwargs={"device": "cpu"}
            )
        
        if embedding_name == EmbeddingType.MULTILP.name:
            # https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
            return SentenceTransformerEmbeddings(
                model_name= 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
            )
        
        raise LlmEmbeddingError(f'Unsupported embedding {embedding_name}')
    
    def get_chunks(self, question : str) -> list[SearchResult]:
        """Find chunks"""

        client = QdrantClient(path = os.path.join(self.__DISK_FOLDER, self.__INDEX_FOLDER))

        qdrant = Qdrant( # pylint: disable=E1102
                    client= client,
                    collection_name= self.__CHUNKS_COLLECTION_NAME,
                    embeddings= self.embeddings
                )

        search_results : list[tuple[Document, float]] = qdrant.similarity_search_with_score(
            question, 
            k= self.sample_count, 
            score_threshold = self.score_threshold
        )
        return [SearchResult(s[0].page_content, s[1], s[0].metadata) for s in search_results]

    def get_answer(self, question : str) -> Answer:
        """Generate answer"""
        return Answer(f'!!{question}!!', 100)
    
    def upload_index(self, uploaded_index_file):
        """Upload index"""
        pass

    def get_query_examples(self) -> list[str]:
        """Get examples of queries, if exists"""
        examples_file_name = os.path.join(self.__DISK_FOLDER, self.__QUERY_EXAMPLES)
        if not os.path.isfile(examples_file_name):
            return ["aaa"]
        with open(examples_file_name, "rt", encoding="utf-8") as f:
            examples = f.readlines()
        examples = [e for e in examples if e and not e.startswith('#')]
        return examples
        

from cohere_ret.cohere_ret import cohere_retriever
from cohere_ret.generator import cohere_generator
from gemini.retrieve import gemini_retriever
from openai_class.retriever import openai_retriever
# from voyageai_ret.retrieve import voyage_retriever
from gemini.generator import gemini_generator
import numpy as np
from chunker.chunk_repo import chunk_repo
from utils.utils import process_duplicates
from utils.utils import FastChunkURLMapper




query = "What are the specific requirements and qualifications needed to issue a good standing certificates for medical staff in sectors where license renewal is fee-exempt?"
ret = gemini_retriever()
chunks = ret.get_Chunks(query)
gen = gemini_generator()
response = gen.generate(query, chunks)



from cohere_ret.cohere_ret import cohere_retriever
from cohere_ret.generator import cohere_generator
from gemini.retrieve import gemini_retriever
from openai_class.retriever import openai_retriever
from voyageai_ret.retrieve import voyage_retriever
from gemini.generator import gemini_generator
import numpy as np
from chunker.chunk_repo import chunk_repo
from utils.utils import process_duplicates
from utils.utils import FastChunkURLMapper
from evaluator.evaluator import RAG_eval




query = "How can a complaint be lodged against a private health facility, and what are the follow-up requirements if the facility’s professional licenses are revoked due to the complaint?"
ret = cohere_retriever()

chunks = ret.get_Chunks(query)
# gen = gemini_generator()
gen = cohere_generator()
# response = gen.generate(query, chunks)

eval = RAG_eval(ret,gen,"gemini-test")
response = eval.query(query)

print(response)



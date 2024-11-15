from cohere_ret.cohere_ret import cohere_retriever
from cohere_ret.generator import cohere_generator
from gemini.retrieve import gemini_retriever
from openai_class.retriever import openai_retriever
# from voyageai_ret.retrieve import voyage_retriever
from gemini.generator import gemini_generator
import numpy as np
from chunker.chunk_repo import chunk_repo
from utils.utils import process_duplicates
from utils.utils import ChunkURLMapper



# repo = chunk_repo("crawled_content.txt")
# print(type(repo.chunks))
# print(len(repo.chunks))
# print(len(list(set(repo.chunks))))

# unique_items, duplicate_items = process_duplicates(repo.chunks)
print("1")
clean_chunks =  chunk_repo("Cleaned_MOHAP.txt").chunks
print("1")

with open('crawled_content.txt', 'r') as f:
    raw_content = f.read()
print("1")


mapper = ChunkURLMapper(clean_chunks, raw_content)
print("1")
mapper.export_to_json('chunk_mapping.json')

from cohere_ret.cohere_ret import cohere_retriever
from cohere_ret.generator import cohere_generator
from gemini.retrieve import gemini_retriever
from openai.retriever import openai_retriever
# from voyageai_ret.retrieve import voyage_retriever
from gemini.generator import gemini_generator
import numpy as np





query = "How can a complaint be lodged against a private health facility, and what are the follow-up requirements if the facility’s professional licenses are revoked due to the complaint?"


retriever = cohere_retriever()
match = retriever.get_Chunks(query)
# # gen = cohere_generator()
# # res = gen.generate(query,match)
# # print("model c:",res.message.content[0].text)

print("model P: ", match)

print("\n")


retriever = openai_retriever()
match = retriever.get_Chunks(query)
print("model R: ", match)


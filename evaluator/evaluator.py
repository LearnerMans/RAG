import numpy as np
import spacy  # Or another library like Gensim

nlp = spacy.load("en_core_web_sm")  # Load a small SpaCy model

def get_sentence_embedding(sentence):
    doc = nlp(sentence)
    return np.mean([token.vector for token in doc], axis=0)

sentence1 = "This is a sentence."
sentence2 = "This is another sentence."

embedding1 = get_sentence_embedding(sentence1)
embedding2 = get_sentence_embedding(sentence2)

# Calculate similarity (e.g., cosine similarity)
similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
print(similarity)
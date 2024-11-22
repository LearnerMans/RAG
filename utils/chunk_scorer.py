from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def calculate_bleu(reference, hypothesis):
    # Apply smoothing to handle zero n-grams gracefully
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference.split()], hypothesis.split(), weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)

def score_retrieved_chunks(query, chunks, ground_truth):
    """
    Evaluates a query's retrieved chunks against a ground truth answer
    using BLEU (2-gram) and ROUGE scores and returns their average.
    
    Parameters:
    - query (str): The query text (not directly used but can be logged or analyzed).
    - chunks (list of str): The list of retrieved chunks to evaluate.
    - ground_truth (str): The ground truth answer text.
    
    Returns:
    - float: The average score of BLEU-2 and ROUGE-L, scaled between 0 and 1.
    """
    if not chunks:
        raise ValueError("The list of chunks is empty.")
    
    # Initialize cumulative scores
    total_bleu = 0
    total_rouge_l = 0

    # Initialize ROUGE scorer
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    # Evaluate each chunk
    for chunk in chunks:
        # BLEU-2 score (unigram + bigram, weights = 0.5 each)
        bleu_score = calculate_bleu(ground_truth, chunk)
        total_bleu += bleu_score

        # ROUGE-L score (recall-oriented metric)
        rouge_scores = rouge.score(ground_truth, chunk)
        rouge_l_score = rouge_scores['rougeL'].fmeasure  # Use F-measure as a balanced score
        total_rouge_l += rouge_l_score

    # Average scores across chunks
    num_chunks = len(chunks)
    average_bleu = total_bleu / num_chunks
    average_rouge_l = total_rouge_l / num_chunks

    # Final average score (scale: 0 to 1)
    final_score = (average_bleu + average_rouge_l) / 2
    print(f"final score {final_score}  , blue {average_bleu} ,  rouge {average_rouge_l}")

    return final_score

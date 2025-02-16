# N-Gram Similarity
def n_gram_similarity(doc1, doc2, n=3):
    def get_ngrams(text, n):
        return [text[i:i+n] for i in range(len(text)-n+1)]

    ngrams1 = set(get_ngrams(doc1, n))
    ngrams2 = set(get_ngrams(doc2, n))

    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))

    return intersection / union if union != 0 else 0
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_tfidf(doc1, doc2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

def cosine_similarity_count(doc1, doc2):
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(count_matrix[0], count_matrix[1])[0][0]

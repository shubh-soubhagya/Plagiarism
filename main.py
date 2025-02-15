from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import numpy as np
import hashlib
from datasketch import MinHash

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Cosine Similarity using TF-IDF
def cosine_similarity_tfidf(doc1, doc2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

# Cosine Similarity using CountVectorizer
def cosine_similarity_count(doc1, doc2):
    vectorizer = CountVectorizer()
    count_matrix = vectorizer.fit_transform([doc1, doc2])
    return cosine_similarity(count_matrix[0], count_matrix[1])[0][0]

# Jaccard Similarity
def jaccard_similarity(doc1, doc2):
    words_doc1 = set(doc1.split())
    words_doc2 = set(doc2.split())
    intersection = words_doc1.intersection(words_doc2)
    union = words_doc1.union(words_doc2)
    return len(intersection) / len(union)

# Longest Common Subsequence (LCS)
def lcs(X, Y):
    m = len(X)
    n = len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n] / max(m, n)

# Locality-Sensitive Hashing (LSH) with MinHash
def lsh_similarity(doc1, doc2, num_perm=128):
    doc1_set = set(doc1.split())
    doc2_set = set(doc2.split())

    minhash1 = MinHash(num_perm=num_perm)
    minhash2 = MinHash(num_perm=num_perm)

    for word in doc1_set:
        minhash1.update(word.encode('utf8'))
    for word in doc2_set:
        minhash2.update(word.encode('utf8'))

    return minhash1.jaccard(minhash2)

# N-Gram Similarity
def n_gram_similarity(doc1, doc2, n=3):
    def get_ngrams(text, n):
        return [text[i:i+n] for i in range(len(text)-n+1)]

    ngrams1 = set(get_ngrams(doc1, n))
    ngrams2 = set(get_ngrams(doc2, n))

    intersection = len(ngrams1.intersection(ngrams2))
    union = len(ngrams1.union(ngrams2))

    return intersection / union if union != 0 else 0

# Example Usage
doc1 = """Renewable energy plays a crucial role in ensuring a sustainable future by reducing dependence on fossil fuels and lowering carbon emissions. Sources such as solar, wind, hydro, and geothermal energy provide eco-friendly alternatives that help combat climate change. Unlike conventional energy sources, renewables generate electricity with minimal environmental impact, making them essential for a greener planet.

Solar energy converts sunlight into electricity through photovoltaic cells, while wind energy harnesses air currents to power turbines. Hydropower utilizes moving water to generate electricity, and geothermal energy relies on the Earth's internal heat for power production. These sustainable energy solutions create employment opportunities, enhance energy security, and promote economic growth.

Despite their advantages, renewable energy sources face challenges such as high installation costs, storage limitations, and reliance on weather conditions. However, advancements in battery technology and smart grids are improving their efficiency and reliability. As governments and industries invest in clean energy, the transition to a sustainable future becomes more feasible. Prioritizing renewable energy adoption will help protect the environment while ensuring long-term energy stability and economic development."""

doc2 = """Renewable energy is essential for creating a sustainable future by reducing fossil fuel consumption and decreasing greenhouse gas emissions. Alternative energy sources like solar, wind, hydropower, and geothermal energy offer environmentally friendly solutions that help mitigate climate change. Unlike traditional energy sources, renewables generate power with minimal ecological harm, making them vital for a cleaner and healthier planet.

Solar power captures sunlight using photovoltaic cells to generate electricity, while wind energy converts air movement into mechanical power. Hydropower generates electricity using flowing water, whereas geothermal energy taps into the Earth's heat for power production. These renewable technologies contribute to job creation, improve energy security, and support economic development.

Although renewable energy offers numerous benefits, it also faces obstacles such as high initial investment, energy storage issues, and dependency on weather conditions. However, technological advancements in battery storage and smart grid systems are enhancing their efficiency and reliability. As governments and corporations continue to invest in sustainable energy, the shift towards a greener future becomes increasingly achievable. Promoting renewable energy adoption will safeguard the environment while ensuring long-term energy security and economic prosperity."""

doc1 = preprocess_text(doc1)
doc2 = preprocess_text(doc2)

print(f"Cosine Similarity (TF-IDF): {cosine_similarity_tfidf(doc1, doc2) * 100:.2f}%")
print(f"Cosine Similarity (CountVectorizer): {cosine_similarity_count(doc1, doc2) * 100:.2f}%")
print(f"Jaccard Similarity: {jaccard_similarity(doc1, doc2) * 100:.2f}%")
print(f"LCS Similarity: {lcs(doc1, doc2) * 100:.2f}%")
print(f"LSH Similarity: {lsh_similarity(doc1, doc2) * 100:.2f}%")
print(f"N-Gram Similarity: {n_gram_similarity(doc1, doc2) * 100:.2f}%")

from plag.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plag.jaccard_similarity import jaccard_similarity
from plag.lcs import lcs
from plag.lsh import lsh_similarity
from plag.n_gram_similarity import n_gram_similarity
from collections import Counter
import re
import numpy as np

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

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

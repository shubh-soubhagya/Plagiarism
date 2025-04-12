import os
import re
from PyPDF2 import PdfReader
from plag.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plag.jaccard_similarity import jaccard_similarity
from plag.lcs import lcs
from plag.lsh import lsh_similarity
from plag.n_gram_similarity import n_gram_similarity
from tabulate import tabulate
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Extract text from PDF
def read_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ''
    for page in reader.pages:
        text += page.extract_text() or ''
    return preprocess_text(text)

# Get all PDF file paths
def get_pdf_files(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]

# Similarity functions
similarity_functions = {
    "Cosine_TFIDF": cosine_similarity_tfidf,
    "Cosine_Count": cosine_similarity_count,
    "Jaccard": jaccard_similarity,
    "LCS": lcs,
    "LSH": lsh_similarity,
    "NGram": n_gram_similarity
}

# Compute similarity for a pair
def compare_pair(i, j, file_names, texts):
    row = {
        "Doc 1": file_names[i],
        "Doc 2": file_names[j]
    }
    scores = []
    for name, func in similarity_functions.items():
        score = round(func(texts[i], texts[j]) * 100, 2)
        row[name] = score
        scores.append(score)
    row["Average Similarity (%)"] = round(np.mean(scores), 2)
    return row

# Main comparison and table generation with threading
def compare_all_pdfs(pdf_folder):
    pdf_files = get_pdf_files(pdf_folder)
    file_names = [os.path.basename(p) for p in pdf_files]

    # Load all PDFs concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        texts = list(executor.map(read_pdf_text, pdf_files))

    results = []

    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(len(pdf_files)):
            for j in range(i + 1, len(pdf_files)):
                futures.append(executor.submit(compare_pair, i, j, file_names, texts))
        for future in futures:
            results.append(future.result())

    print(tabulate(results, headers="keys", tablefmt="grid"))

# Run it
compare_all_pdfs("pdf_app_test")



# import os
# import re
# from PyPDF2 import PdfReader
# from plag.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
# from plag.jaccard_similarity import jaccard_similarity
# from plag.lcs import lcs
# from plag.lsh import lsh_similarity
# from plag.n_gram_similarity import n_gram_similarity
# from tabulate import tabulate
# import numpy as np

# # Preprocessing Function
# def preprocess_text(text):
#     text = text.lower()
#     text = re.sub(r'[^\w\s]', '', text)
#     return text

# # Extract text from PDF
# def read_pdf_text(pdf_path):
#     reader = PdfReader(pdf_path)
#     text = ''
#     for page in reader.pages:
#         text += page.extract_text() or ''
#     return preprocess_text(text)

# # Get all PDF file paths
# def get_pdf_files(folder_path):
#     return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]

# # Similarity functions
# similarity_functions = {
#     "Cosine_TFIDF": cosine_similarity_tfidf,
#     "Cosine_Count": cosine_similarity_count,
#     "Jaccard": jaccard_similarity,
#     "LCS": lcs,
#     "LSH": lsh_similarity,
#     "NGram": n_gram_similarity
# }

# # Main comparison and table generation
# def compare_all_pdfs(pdf_folder):
#     pdf_files = get_pdf_files(pdf_folder)
#     file_names = [os.path.basename(p) for p in pdf_files]
#     texts = [read_pdf_text(p) for p in pdf_files]

#     results = []

#     for i in range(len(pdf_files)):
#         for j in range(i + 1, len(pdf_files)):
#             row = {
#                 "Doc 1": file_names[i],
#                 "Doc 2": file_names[j]
#             }
#             scores = []
#             for name, func in similarity_functions.items():
#                 score = round(func(texts[i], texts[j]) * 100, 2)
#                 row[name] = score
#                 scores.append(score)
#             row["Average Similarity (%)"] = round(np.mean(scores), 2)
#             results.append(row)

#     headers = ["Doc 1", "Doc 2"] + list(similarity_functions.keys()) + ["Average Similarity (%)"]
#     print(tabulate(results, headers="keys", tablefmt="grid"))

# # Run it
# compare_all_pdfs("pdf_app_test")

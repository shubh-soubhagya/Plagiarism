import os
import re
import json
import base64
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from plag.cosine_similarity import cosine_similarity_count, cosine_similarity_tfidf
from plag.jaccard_similarity import jaccard_similarity
from plag.lcs import lcs
from plag.lsh import lsh_similarity
from plag.n_gram_similarity import n_gram_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import tempfile
import shutil
import csv
import io
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload

# Preprocessing Function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Extract text from PDF
def read_pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return preprocess_text(text)
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return ""

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
        try:
            score = round(func(texts[i], texts[j]) * 100, 2)
        except Exception as e:
            print(f"Error computing {name} for {file_names[i]} and {file_names[j]}: {e}")
            score = 0.0
        row[name] = score
        scores.append(score)
    row["Average Similarity (%)"] = round(np.mean(scores), 2)
    return row

# Save uploaded files and process them
def process_uploaded_pdfs(files):
    # Create a temporary directory to store the uploaded PDFs
    temp_dir = tempfile.mkdtemp()
    file_paths = []
    
    try:
        # Save the files to the temporary directory
        for file in files:
            if file.filename.lower().endswith('.pdf'):
                file_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.save(file_path)
                file_paths.append(file_path)
        
        # Process the saved files
        results = compare_pdfs(file_paths)
        return results
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir)

# Main comparison function
def compare_pdfs(pdf_files):
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

    return results

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for processing PDFs
@app.route('/api/process', methods=['POST'])
def process_pdfs():
    if 'files[]' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files[]')
    if not files or len(files) < 2:
        return jsonify({"error": "Please upload at least 2 PDF files"}), 400
    
    # Filter for PDF files only
    pdf_files = [f for f in files if f.filename.lower().endswith('.pdf')]
    if len(pdf_files) < 2:
        return jsonify({"error": "Please upload at least 2 PDF files"}), 400
    
    try:
        results = process_uploaded_pdfs(pdf_files)
        return jsonify({
            "results": results, 
            "fileCount": len(pdf_files),
            "fileNames": [f.filename for f in pdf_files]
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API endpoint for downloading results as CSV
@app.route('/api/download/csv', methods=['POST'])
def download_csv():
    try:
        data = request.json
        results = data.get('results', [])
        
        if not results:
            return jsonify({"error": "No results to download"}), 400
        
        # Create a CSV in memory
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
        # Create a temporary file to save the CSV
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        with open(temp_file.name, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        return send_file(temp_file.name, 
                         mimetype='text/csv',
                         as_attachment=True, 
                         download_name=f'similarity_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Generate HTML report content
def generate_html_report(results, file_names):    
    report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    table_rows = ""
    for result in results:
        table_rows += f"""
        <tr>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["Doc 1"]}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["Doc 2"]}</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["Cosine_TFIDF"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["Cosine_Count"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["Jaccard"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["LCS"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["LSH"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px;">{result["NGram"]}%</td>
            <td style="border: 1px solid #ddd; padding: 8px; font-weight: bold;">{result["Average Similarity (%)"]}%</td>
        </tr>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document Similarity Report - {report_date}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2563eb; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th {{ background-color: #f1f5f9; text-align: left; padding: 12px; border: 1px solid #ddd; }}
            td {{ padding: 8px; border: 1px solid #ddd; }}
            tr:nth-child(even) {{ background-color: #f9fafb; }}
            .summary {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Document Similarity Analysis Report</h1>
        <div class="summary">
            <p><strong>Generated on:</strong> {report_date}</p>
            <p><strong>Files analyzed:</strong> {len(file_names)}</p>
            <p><strong>File names:</strong> {', '.join(file_names)}</p>
        </div>
        
        <h2>Similarity Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Doc 1</th>
                    <th>Doc 2</th>
                    <th>Cosine_TFIDF (%)</th>
                    <th>Cosine_Count (%)</th>
                    <th>Jaccard (%)</th>
                    <th>LCS (%)</th>
                    <th>LSH (%)</th>
                    <th>NGram (%)</th>
                    <th>Average (%)</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <h2>Metrics Explanation</h2>
        <table>
            <thead>
                <tr>
                    <th style="width: 25%;">Metric</th>
                    <th style="width: 75%;">Explanation</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>Cosine TF-IDF</strong></td>
                    <td>Measures similarity by treating documents as vectors, with words weighted by their importance. Higher scores indicate documents share important terms, not just common words.</td>
                </tr>
                <tr>
                    <td><strong>Cosine Count</strong></td>
                    <td>Similar to Cosine TF-IDF but uses raw word counts instead of weighted values. Measures how similar the word distributions are between documents.</td>
                </tr>
                <tr>
                    <td><strong>Jaccard</strong></td>
                    <td>Compares the shared words between documents to the total unique words in both. Focuses on word overlap regardless of frequency or order.</td>
                </tr>
                <tr>
                    <td><strong>LCS</strong></td>
                    <td>Finds the longest sequence of words that appear in the same order in both documents. Good for detecting large blocks of identical text.</td>
                </tr>
                <tr>
                    <td><strong>LSH</strong></td>
                    <td>Uses hashing to quickly identify similar document segments. Effective for detecting partial matches and document sections that have been copied.</td>
                </tr>
                <tr>
                    <td><strong>NGram</strong></td>
                    <td>Compares sequences of consecutive words (typically 3-5 words) between documents. Good for identifying phrase-level similarity and paraphrasing.</td>
                </tr>
                <tr>
                    <td><strong>Average</strong></td>
                    <td>The mean of all similarity metrics, giving an overall indication of document similarity. Higher percentages suggest a greater likelihood of content overlap.</td>
                </tr>
            </tbody>
        </table>
        
        <div style="margin-top: 30px; color: #6b7280; font-size: 12px; text-align: center;">
            <p>Generated by DocSimilarity - PDF Document Similarity Analysis Tool</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

# API endpoint for downloading results as HTML report
@app.route('/api/download/html', methods=['POST'])
def download_html():
    try:
        data = request.json
        results = data.get('results', [])
        file_names = data.get('fileNames', [])
        
        if not results:
            return jsonify({"error": "No results to download"}), 400
        
        # Generate HTML report
        html_content = generate_html_report(results, file_names)
        
        # Create a temporary file to save the HTML report
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        with open(temp_file.name, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return send_file(temp_file.name, 
                         mimetype='text/html',
                         as_attachment=True, 
                         download_name=f'similarity_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Make sure templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Check if index.html exists, if not warn the user
    if not os.path.exists('templates/index.html'):
        print("\nWARNING: templates/index.html not found!")
        print("Please create the file 'templates/index.html' with the HTML content provided earlier.\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
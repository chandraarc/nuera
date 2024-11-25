
import numpy as np
import os
import sys
from sentence_transformers import SentenceTransformer, util
import PyPDF2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        num_pages = len(pdf_reader.pages)
        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text = text+"  "+(""+page.extract_text()).replace("\n", "")

        print(text)
        return str(text).split("  ")

def load_documents_from_directory(directory_path, documents):

    finalvals=[]
    for filename in os.listdir(directory_path):
        document = extract_text_from_pdf(directory_path+"/"+filename)
        documents= np.concatenate([documents, document])

    for sentence in documents:
        if len(sentence.strip()) > 1:
            finalvals.append(sentence.strip().lower())

    return finalvals;

# Example usage
directory_path = "/home/chandra/mlsampledocuments"  # Replace with your desired model
documents = []
finalvals = load_documents_from_directory(directory_path, documents)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(finalvals)
while True:
    name = sys.stdin.readline().strip()
    text_vector = vectorizer.transform([name.lower()])

    vocabulary = vectorizer.get_feature_names_out()

    similarities = cosine_similarity(text_vector, tfidf_matrix)
    print(similarities)
    similarity_threshold = 0.6
    max_value = np.max(similarities[0])
    max_index = np.argmax(similarities[0])
    print(finalvals[max_index]+finalvals[max_index+1]+finalvals[max_index+2])

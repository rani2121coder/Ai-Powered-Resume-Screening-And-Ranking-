import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import PyPDF2

# Custom list of stopwords
CUSTOM_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for",
    "with", "about", "against", "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
}

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    # Tokenize text using split()
    tokens = text.split()
    # Remove stopwords using custom list
    tokens = [word for word in tokens if word not in CUSTOM_STOPWORDS]
    return ' '.join(tokens)

# Function to rank resumes
def rank_resumes(job_description, resumes):
    # Preprocess job description
    job_description_processed = preprocess_text(job_description)
    
    # Preprocess resumes
    resumes_processed = [preprocess_text(resume) for resume in resumes]
    
    # Combine job description and resumes for vectorization
    all_texts = [job_description_processed] + resumes_processed
    
    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Calculate cosine similarity between job description and resumes
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Rank resumes based on similarity scores
    ranked_indices = cosine_similarities.argsort()[::-1]
    ranked_resumes = [(resumes[i], cosine_similarities[i]) for i in ranked_indices]
    
    return ranked_resumes

# Streamlit App
def main():
    st.title("AI-Powered Resume Screening and Ranking System")
    
    # Input job description
    job_description = st.text_area("Enter the Job Description:", height=200)
    
    # Upload resumes
    uploaded_files = st.file_uploader("Upload Resumes (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("Rank Resumes"):
        if job_description and uploaded_files:
            resumes = []
            for uploaded_file in uploaded_files:
                if uploaded_file.type == "application/pdf":
                    # Extract text from PDF using PdfReader
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text()
                    resumes.append(text)
                elif uploaded_file.type == "text/plain":
                    # Read text from TXT file
                    text = uploaded_file.read().decode("utf-8")
                    resumes.append(text)
            
            # Rank resumes
            ranked_resumes = rank_resumes(job_description, resumes)
            
            # Display ranked resumes
            st.subheader("Ranked Resumes:")
            for i, (resume, score) in enumerate(ranked_resumes):
                st.write(f"Rank {i+1}: Similarity Score = {score:.4f}")
                st.text_area(f"Resume {i+1}", resume, height=200)
        else:
            st.warning("Please provide both a job description and at least one resume.")

if __name__ == "__main__":
    main()
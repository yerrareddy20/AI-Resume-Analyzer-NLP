import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)

    # Remove punctuation and stopwords
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stopwords.words('english')]

    return " ".join(tokens)

# -------------------------------
# Input Resume and Job Description
# -------------------------------
resume = input("Enter Resume Text:\n")
job_desc = input("\nEnter Job Description:\n")

# Preprocess
resume_clean = preprocess_text(resume)
job_clean = preprocess_text(job_desc)

# -------------------------------
# TF-IDF Vectorization
# -------------------------------
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([resume_clean, job_clean])

# -------------------------------
# Similarity Calculation
# -------------------------------
similarity_score = cosine_similarity(vectors[0], vectors[1])[0][0]

# -------------------------------
# Output
# -------------------------------
print("\n--- RESULT ---")
print(f"Similarity Score: {round(similarity_score * 100, 2)} %")

if similarity_score > 0.7:
    print("Strong Match ✅")
elif similarity_score > 0.4:
    print("Moderate Match ⚠️")
else:
    print("Low Match ❌")
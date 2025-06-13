from nlp_utils.preprocessor import preprocess
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import numpy as np
from collections import defaultdict
import spacy


# Read data from "data.txt"
with open("data.txt" , "r", encoding="utf-8") as file:
    text = file.read()

# Preprocess text
# preprocessed_text = preprocess(text)

# Sentence Tokenizer
# sentences  = sent_tokenize(preprocessed_text)
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
sentences = [sent.text.strip() for sent in doc.sents]


print(f"Total sentences extracted: {len(sentences)}")


# Generating Sentence Embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(sentences, convert_to_tensor=True)

# Cluster semantically similar questions
clustering = DBSCAN(eps=0.7, min_samples=2, metric='cosine')
labels = clustering.fit_predict(embeddings.cpu().numpy())

print(f"Cluster labels: {labels}")


# Group and count questions by cluster
clusters = defaultdict(list)
for sentence, label in zip(sentences, labels):
    if label != -1:  # -1 is noise in DBSCAN
        clusters[label].append(sentence)

# Show Frequency - Semantic Count
for cluster_id, questions in clusters.items():
    print(f"Cluster {cluster_id} (Frequency: {len(questions)})")
    for q in questions:
        print("  -", q)

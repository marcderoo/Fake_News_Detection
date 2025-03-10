from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

# Charger Word2Vec une seule fois
print("[INFO] Loading Word2Vec model (word2vec-google-news-300)...")
word2vec_model = api.load("word2vec-google-news-300")
print("[INFO] Word2Vec model loaded successfully!")

# Charger BERT une seule fois
print("[INFO] Loading BERT model (bert-base-uncased)...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
print("[INFO] BERT model loaded successfully!")

def extract_tfidf_features(train_texts, test_texts, max_features=5000):
    """Extrait les caractéristiques TF-IDF et retourne aussi le vectorizer."""
    print("[INFO] Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    print("[INFO] TF-IDF extraction complete!")
    return X_train, X_test, vectorizer

def extract_word2vec_features(train_texts, test_texts):
    """Extrait les caractéristiques Word2Vec."""
    print("[INFO] Extracting Word2Vec features...")

    def get_vector(text):
        words = text.split()
        vectors = [word2vec_model[word] for word in words if word in word2vec_model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(word2vec_model.vector_size)

    X_train = np.array([get_vector(text) for text in train_texts])
    X_test = np.array([get_vector(text) for text in test_texts])
    
    print("[INFO] Word2Vec extraction complete!")
    return X_train, X_test

def extract_bert_features(train_texts, test_texts):
    """Extrait les embeddings BERT."""
    print("[INFO] Extracting BERT features...")

    def get_bert_embedding(text):
        if not isinstance(text, str) or len(text) == 0:
            return np.zeros(bert_model.config.hidden_size)  
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy(force=True)  

    X_train = np.array([get_bert_embedding(text) for text in train_texts])
    X_test = np.array([get_bert_embedding(text) for text in test_texts])

    print("[INFO] BERT extraction complete!")
    return X_train, X_test

# Test si on exécute directement le fichier
if __name__ == "__main__":
    from data_loader import load_isot_dataset

    print("[INFO] Loading dataset for feature extraction test...")
    train_data, test_data = load_isot_dataset("data/Fake.csv", "data/True.csv", preprocessing_type="tfidf")

    print("[INFO] Running TF-IDF extraction test...")
    X_train_tfidf, X_test_tfidf, _ = extract_tfidf_features(train_data["text"], test_data["text"])

    print("[INFO] Running Word2Vec extraction test...")
    X_train_w2v, X_test_w2v = extract_word2vec_features(train_data["text"], test_data["text"])

    print("[INFO] Running BERT extraction test...")
    X_train_bert, X_test_bert = extract_bert_features(train_data["text"], test_data["text"])

    print("[INFO] Feature extraction testing complete!")
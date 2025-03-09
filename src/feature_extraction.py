from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

def extract_tfidf_features(train_texts, test_texts, max_features=5000):
    """Extracts TF-IDF features."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test

def extract_word2vec_features(train_texts, test_texts, model_name="word2vec-google-news-300"):
    """Extracts Word2Vec features."""
    word2vec_model = api.load(model_name)

    def get_vector(text):
        words = text.split()
        vectors = [word2vec_model[word] for word in words if word in word2vec_model]
        return np.mean(vectors, axis=0) if vectors else np.zeros(300)

    X_train = np.array([get_vector(text) for text in train_texts])
    X_test = np.array([get_vector(text) for text in test_texts])
    return X_train, X_test

def extract_bert_features(train_texts, test_texts, model_name="bert-base-uncased"):
    """Extracts BERT embeddings."""
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    def get_bert_embedding(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    X_train = np.array([get_bert_embedding(text) for text in train_texts])
    X_test = np.array([get_bert_embedding(text) for text in test_texts])
    return X_train, X_test
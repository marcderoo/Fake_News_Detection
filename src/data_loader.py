import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))  # Ensemble pour une recherche rapide
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_text_tfidf(text):
    """Prétraitement complet pour TF-IDF et BoW : suppression agressive."""
    if not isinstance(text, str) or len(text) == 0:
        return ""

    text = text.lower()  
    text = re.sub(r"http\S+", "", text)  # Supprimer les URLs
    text = re.sub(r"@\S+", "", text)  # Supprimer les handles Twitter
    text = re.sub(r"[^a-zA-Z]", " ", text)  # Supprimer tout sauf les lettres
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in STOPWORDS])  # Lemmatisation + suppression stopwords

    return text

def clean_text_embeddings(text):
    """Prétraitement léger pour Word2Vec et BERT : on garde la structure."""
    if not isinstance(text, str) or len(text) == 0:
        return ""

    text = text.lower()  
    text = re.sub(r"http\S+", "", text)  # Supprimer les URLs
    text = re.sub(r"@\S+", "", text)  # Supprimer les handles Twitter
    text = re.sub(r"\s+", " ", text).strip()  # Supprimer espaces multiples

    return text

def load_isot_dataset(fake_path, real_path, preprocessing_type="tfidf"):
    """Charge le dataset ISOT avec le bon type de preprocessing."""
    print("[INFO] Loading dataset...")

    df_fake = pd.read_csv(fake_path)
    df_real = pd.read_csv(real_path)

    df_fake["label"] = 0
    df_real["label"] = 1

    df = pd.concat([df_fake, df_real]).reset_index(drop=True)

    print(f"[INFO] Applying {preprocessing_type.upper()} preprocessing...")
    if preprocessing_type == "tfidf":
        df["text"] = df["text"].apply(clean_text_tfidf)
    elif preprocessing_type == "embeddings":
        df["text"] = df["text"].apply(clean_text_embeddings)
    else:
        raise ValueError("preprocessing_type must be 'tfidf' or 'embeddings'")

    print("[INFO] Splitting dataset...")
    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    print("[INFO] Data loading complete!")
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = load_isot_dataset("data/Fake.csv", "data/True.csv", preprocessing_type="tfidf")
    print(train_data.head())
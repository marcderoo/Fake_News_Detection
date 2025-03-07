import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')

def clean_text(text):
    """Nettoyage des textes : suppression des URLs, ponctuation et stopwords."""
    text = re.sub(r"http\S+", "", text)  # Supprime les liens
    text = text.translate(str.maketrans("", "", string.punctuation))  # Supprime la ponctuation
    text = text.lower()  # Convertit en minuscules
    text = " ".join([word for word in text.split() if word not in stopwords.words('english')])  # Supprime les stopwords
    return text

def load_isot_dataset(path_fake, path_real):
    """Charge et nettoie le dataset ISOT."""
    print("[INFO] Chargement des fichiers...")
    df_fake = pd.read_csv(path_fake)
    df_real = pd.read_csv(path_real)

    print("[INFO] Assignation des labels...")
    df_fake["label"] = 0  # Fake news = 0
    df_real["label"] = 1  # True news = 1

    df = pd.concat([df_fake, df_real]).reset_index(drop=True)
    print(f"[INFO] Nombre total d'articles : {len(df)}")

    print("[INFO] Nettoyage des textes...")
    # df["text"] = df["text"].apply(clean_text)  # Appliquer le nettoyage
    print("[INFO] Nettoyage terminé.")

    # Séparer en train/test (80/20)
    print("[INFO] Séparation train/test...")
    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    print("[INFO] Données prêtes.")
    
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = load_isot_dataset("data/Fake.csv", "data/True.csv")
    print(train_data.head())
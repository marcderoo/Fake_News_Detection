import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Download stopwords once
nltk.download('stopwords')
STOPWORDS = set(stopwords.words('english'))  # Use set for faster lookup

def clean_text(text):
    """Fast text preprocessing: removes punctuation, stopwords, and converts to lowercase."""
    if not isinstance(text, str) or len(text) == 0:  
        return ""  # Handle NaN or empty text cases
    
    text = text.lower()  # Lowercase once
    text = text.replace("\n", " ")  # Remove newlines
    text = text.translate(str.maketrans("", "", string.punctuation))  # Fast punctuation removal
    words = text.split()  # Tokenize fast
    return " ".join([word for word in words if word not in STOPWORDS])  # Stopword removal

def load_isot_dataset(fake_path, real_path, sample_size=None):
    """Loads and preprocesses ISOT dataset efficiently."""
    print("[INFO] Loading dataset...")

    df_fake = pd.read_csv(fake_path)
    df_real = pd.read_csv(real_path)

    df_fake["label"] = 0  
    df_real["label"] = 1  

    df = pd.concat([df_fake, df_real]).reset_index(drop=True)

    # Optional: Speed up testing by using a subset
    if sample_size:
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)

    print("[INFO] Preprocessing text (fast mode)...")
    df["text"] = df["text"].astype(str)  # Ensure all texts are strings
    df["text"] = df["text"].apply(clean_text)  # Vectorized cleaning

    print("[INFO] Splitting dataset...")
    train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    print("[INFO] Data loading complete!")
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = load_isot_dataset("data/Fake.csv", "data/True.csv", sample_size=5000)  # Use a sample for testing
    print(train_data.head())
import sys
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from feature_extraction import extract_tfidf_features  # Ou autre méthode utilisée
from data_loader import load_isot_dataset

def evaluate_model(X_test, y_test, model_path):
    """Évalue un modèle chargé depuis un fichier."""
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"Erreur : fichier {model_path} introuvable.")
        sys.exit(1)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage : python evaluate.py <model_path>")
        sys.exit(1)

    model_name = sys.argv[1]  # Exemple : "models/svm.pkl"

    # Charger les données de test
    print("[INFO] Chargement des données...")
    train_data, test_data = load_isot_dataset("data/Fake.csv", "data/True.csv")
    X_train, X_test = extract_tfidf_features(train_data["text"], test_data["text"])
    y_train, y_test = train_data["label"], test_data["label"]

    print("[INFO] Évaluation du modèle...")
    evaluate_model(X_test, y_test, model_name)
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from feature_extraction import extract_tfidf_features
from data_loader import load_isot_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Charger le dataset avec preprocessing TF-IDF
print("[INFO] Loading dataset...")
train_data, test_data = load_isot_dataset("data/Fake.csv", "data/True.csv", preprocessing_type="tfidf")

# Extraire les caractéristiques TF-IDF
X_train_tfidf, X_test_tfidf, vectorizer = extract_tfidf_features(train_data["text"], test_data["text"])
y_train, y_test = train_data["label"], test_data["label"]

# Sauvegarde du vectorizer pour l'évaluation
os.makedirs("models", exist_ok=True)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
print("[INFO] TF-IDF vectorizer saved!")

# Définition des modèles utilisés dans l'article
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel="linear"),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100),
    "AdaBoost": AdaBoostClassifier(n_estimators=100),
    "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
}

# Entraînement et évaluation immédiate
for model_name, model in models.items():
    print(f"\n[INFO] Training {model_name}...")
    model.fit(X_train_tfidf, y_train)
    
    # Évaluer immédiatement avant la sauvegarde
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"[INFO] {model_name} Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Sauvegarde du modèle
    with open(f"models/{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] {model_name} saved to models/{model_name}.pkl")

print("[INFO] All models trained and saved successfully!")
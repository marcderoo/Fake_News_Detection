import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from data_loader import load_isot_dataset
from feature_extraction import extract_tfidf_features

# Load and preprocess data
print("[INFO] Loading dataset...")
train_data, test_data = load_isot_dataset("data/Fake.csv", "data/True.csv")

print("[INFO] Extracting TF-IDF features...")
X_train, X_test = extract_tfidf_features(train_data["text"], test_data["text"])
y_train, y_test = train_data["label"], test_data["label"]

# Dictionary of models
models = {
    "SVM": SVC(kernel="linear"),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "LogisticRegression": LogisticRegression(),
    "NaiveBayes": MultinomialNB()
}

# Train each model and save it
for model_name, model in models.items():
    print(f"[INFO] Training {model_name}...")
    model.fit(X_train, y_train)
    print(f"[INFO] {model_name} training complete!")

    # Save the model
    model_path = f"models/{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"[INFO] {model_name} saved to {model_path}.")

print("[INFO] All models trained and saved successfully!")
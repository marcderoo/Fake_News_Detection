import pickle
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from feature_extraction import extract_tfidf_features, extract_word2vec_features
from data_loader import load_isot_dataset
from train_DL import LSTMClassifier

# Load dataset
print("[INFO] Loading test dataset...")
_, test_data = load_isot_dataset("data/Fake.csv", "data/True.csv")

# Extract features for ML models
print("[INFO] Extracting TF-IDF features for ML models...")
_, X_test_tfidf = extract_tfidf_features(test_data["text"], test_data["text"])
y_test = test_data["label"]

# Extract features for Deep Learning models
print("[INFO] Extracting Word2Vec embeddings for LSTM...")
_, X_test_word2vec = extract_word2vec_features(test_data["text"], test_data["text"])

# Convert to PyTorch tensors
X_test_word2vec = torch.tensor(X_test_word2vec, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

# List of trained ML models
ml_models = ["SVM", "RandomForest", "LogisticRegression", "NaiveBayes"]

# Evaluate ML models
for model_name in ml_models:
    print(f"\n[INFO] Evaluating {model_name}...")
    
    # Load the model
    with open(f"models/{model_name}.pkl", "rb") as f:
        model = pickle.load(f)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Print results
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Evaluate LSTM Model
print("\n[INFO] Evaluating LSTM Model...")

# Load LSTM model
input_size = X_test_word2vec.shape[1]
hidden_size = 128
num_classes = 2
lstm_model = LSTMClassifier(input_size, hidden_size, num_classes)
lstm_model.load_state_dict(torch.load("models/lstm.pth"))
lstm_model.eval()

# Make predictions
with torch.no_grad():
    outputs = lstm_model(X_test_word2vec)
    y_pred_lstm = torch.argmax(outputs, dim=1).numpy()

# Print results
print(f"Accuracy: {accuracy_score(y_test, y_pred_lstm):.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred_lstm))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_lstm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.title("Confusion Matrix - LSTM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
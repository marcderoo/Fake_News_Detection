from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import os

def train_model(X_train, y_train, model_type="svm"):
    """Entraîne un modèle classique et sauvegarde le modèle."""
    if model_type == "svm":
        model = SVC(kernel="linear")
    elif model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=100)
    elif model_type == "logistic_regression":
        model = LogisticRegression()

    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)

    # Sauvegarde du modèle
    with open(f"models/{model_type}.pkl", "wb") as f:
        pickle.dump(model, f)

    return model
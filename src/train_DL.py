import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from feature_extraction import extract_word2vec_features, extract_bert_features
from data_loader import load_isot_dataset

# -------------- LSTM CLASSIFIER --------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  
        out = self.fc(hn[-1])  
        return out  

# -------------- BERT CLASSIFIER --------------
class BERTClassifier(nn.Module):
    def __init__(self, bert_model="bert-base-uncased", num_classes=2):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)  

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        out = self.fc(outputs.pooler_output)  
        return out  

# -------------- LOAD DATA --------------
print("[INFO] Loading dataset...")
train_data, test_data = load_isot_dataset("data/Fake.csv", "data/True.csv", preprocessing_type="embeddings")

print("[INFO] Extracting Word2Vec embeddings for LSTM...")
X_train_w2v, X_test_w2v = extract_word2vec_features(train_data["text"], test_data["text"])
y_train, y_test = train_data["label"].values, test_data["label"].values

print("[INFO] Extracting BERT embeddings...")
X_train_bert, X_test_bert = extract_bert_features(train_data["text"], test_data["text"])

# Convert to PyTorch tensors
X_train_w2v, X_test_w2v = torch.tensor(X_train_w2v, dtype=torch.float32), torch.tensor(X_test_w2v, dtype=torch.float32)
y_train_tensor, y_test_tensor = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# -------------- TRAINING FUNCTION --------------
def train_model(model, train_loader, epochs=3, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)  
            labels = labels.view(-1)  # Ensure labels are 1D
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"[INFO] Epoch {epoch+1} completed. Avg Loss: {total_loss/len(train_loader):.4f}")

# -------------- TRAIN LSTM --------------
print("[INFO] Training LSTM Model...")
lstm_model = LSTMClassifier(input_size=X_train_w2v.shape[1], hidden_size=128, num_classes=2)
train_dataset = TensorDataset(X_train_w2v, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
train_model(lstm_model, train_loader)
torch.save(lstm_model.state_dict(), "models/lstm.pth")
print("[INFO] LSTM Model Saved!")

# -------------- TRAIN BERT --------------
print("[INFO] Training BERT Model...")
bert_model = BERTClassifier()
train_dataset = TensorDataset(X_train_bert["input_ids"], X_train_bert["attention_mask"], y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(bert_model.parameters(), lr=2e-5)

for epoch in range(3):
    total_loss = 0
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = bert_model(input_ids, attention_mask)  
        labels = labels.view(-1)  
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 5 == 0:
            print(f"BERT Epoch [{epoch+1}/3], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"[INFO] BERT Epoch {epoch+1} completed. Avg Loss: {total_loss/len(train_loader):.4f}")

torch.save(bert_model.state_dict(), "models/bert.pth")
print("[INFO] BERT Model Saved!")
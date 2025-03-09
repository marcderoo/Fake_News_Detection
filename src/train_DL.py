import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from feature_extraction import extract_word2vec_features
from data_loader import load_isot_dataset

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # Ensure output has shape [batch_size, num_classes]

    def forward(self, x):
        _, (hn, _) = self.lstm(x)  # Get last hidden state
        out = self.fc(hn[-1])  # Linear layer maps to num_classes
        return out  # Output shape: [batch_size, num_classes]

# Load data
print("[INFO] Loading dataset...")
train_data, test_data = load_isot_dataset("data/Fake.csv", "data/True.csv", sample_size=5000)  # Use sample for speed

print("[INFO] Extracting Word2Vec embeddings...")
X_train, X_test = extract_word2vec_features(train_data["text"], test_data["text"])
y_train, y_test = train_data["label"].values, test_data["label"].values

# Convert to PyTorch tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

# Training parameters
input_size = X_train.shape[1]
hidden_size = 128
num_classes = 2
epochs = 5
batch_size = 32
lr = 0.001

# Initialize model
print("[INFO] Initializing LSTM model...")
model = LSTMClassifier(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Prepare DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
print("[INFO] Starting training...")
for epoch in range(epochs):
    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad() 
        outputs = model(inputs)  # Shape: [batch_size, num_classes]
        labels = labels.view(-1)  # Ensure labels are 1D: shape [batch_size]
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % 10 == 0:  # Print progress every 10 batches
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    print(f"[INFO] Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_loader):.4f}")

# Save the trained model
torch.save(model.state_dict(), "models/lstm.pth")
print("[INFO] LSTM model training complete and saved to models/lstm.pth!")
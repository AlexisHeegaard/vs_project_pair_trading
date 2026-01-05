# src/models.py
#PROJECT BLUEPRINT: ADVANCED MODELS FOR PAIR TRADING

import torch
import torch.nn as nn
#those libraries contain building blocks we need to build deep learning methods
from sklearn.linear_model import RidgeClassifier
#ridgeclassifier is imported to create the baseline model

# TIER 1: THE BASELINE (Ridge Regression) 
#models mean reversion as a linear process
class BaselineModel:
    def __init__(self, alpha=1.0):
        self.model = RidgeClassifier(alpha=alpha)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)


# TIER 2: CLASSICAL ML (Multi-Layer Perceptron)
#standard "feed-forward neural network"
#captures non-linear interactions effects
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU() # Non-linear activation
        self.dropout = nn.Dropout(0.2) # Prevents overfitting
        self.layer2 = nn.Linear(hidden_dim, 1) 
        self.sigmoid = nn.Sigmoid() # Outputs probability (0 to 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return self.sigmoid(x)


# TIER 3: DEEP LEARNING (LSTM)
#captures temporal dependencies in data
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # The LSTM Layer (The "Memory" Engine)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # The Output Layer (Decision Maker)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch_Size, Sequence_Length, Features)
        
        # Initialize hidden state (h0) and cell state (c0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the last time step (Today's prediction)
        out = out[:, -1, :] 
        
        out = self.fc(out)
        return self.sigmoid(out)
    

    
#tier 3.5: deep learning (GRU)
# captures temporal dependencies in data, similar to LSTM but with a different architecture
# often more efficient than LSTM
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        """
        Tier 3.5: GRU (Gated Recurrent Unit).
        Simpler than LSTM (fewer parameters), often faster and 
        less prone to overfitting on noisy financial data.
        """
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU Layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Output Layer
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Initialize hidden state (h0)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate GRU
        # GRU does not use cell state (c0), unlike LSTM
        out, _ = self.gru(x, h0)
        
        # Take the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)   
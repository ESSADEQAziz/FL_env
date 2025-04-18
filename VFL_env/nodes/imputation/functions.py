import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class SimpleRegressor(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        super(SimpleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class ClientEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=4):
        super(ClientEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

def preprocess_client_data(csv_path,drop_list):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=drop_list, errors="ignore")

    num_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = df.select_dtypes(include=['object']).columns.tolist()

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    X = preprocessor.fit_transform(df)
    return torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32)

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
    
# class ClientLinearEncoder(nn.Module):
#     def __init__(self, input_dim):
#         super(ClientEncoder, self).__init__()
#         self.linear = nn.Linear(input_dim, input_dim)  # Identity-like map

#     def forward(self, x):
#         return self.linear(x)
    

def preprocess_client_data(csv_path,target_features):
    df = pd.read_csv(csv_path)
    column_names = df.columns

    num_features=[]
    cat_features=[]

    for item in column_names:
        if item in target_features :
            if df[item].dtype in ['int64', 'float64'] :
                num_features.append(item)
                pass
            elif df[item].dtype in ['object']:
                cat_features.append(item)
                pass

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    X = preprocessor.fit_transform(df)

    return torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32)




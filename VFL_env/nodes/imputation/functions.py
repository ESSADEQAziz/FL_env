import torch
import torch.nn as nn
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class ClientEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim=4):
        super(ClientEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)
    
def preprocess_features(csv_path,target_features,approche):
    df = pd.read_csv(csv_path)
    column_names = df.columns

    num_features=[]
    cat_features=[]

    for item in column_names:
        if item in target_features :
            if df[item].dtype in ['int64', 'float64'] :
                num_features.append(item)
            elif df[item].dtype in ['object']:
                cat_features.append(item)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # Save the features processor for the future tests and evaluations 
    if approche == 'dl_r':
        preprocessor_path = f"../results/dl_regression/"
    elif approche == 'dl_c':
        preprocessor_path = f"../results/dl_classification/"
    elif approche == 'ml_r':
        preprocessor_path = f"../results/ml_regression/"
    elif approche == 'ml_c':
        preprocessor_path = f"../results/ml_classification/"
      
    os.makedirs(preprocessor_path, exist_ok=True)
    X = preprocessor.fit_transform(df)
    with open(f"{preprocessor_path}preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    return torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32)

def insure_none(x):
    if torch.isnan(x).any():
        print("Warning: NaN values detected in target y, replacing them with 0 to avoid loss and gradients calculus.(consider it as noise)")
        x = torch.nan_to_num(x, nan=0.0)
    return x 





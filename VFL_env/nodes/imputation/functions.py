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
    
def preprocess_node_data_ml(csv_path,target_features,approche):
    df = pd.read_csv(csv_path)
    column_names = df.columns

    num_features=[]
    cat_features=[]

    if approche == "regression":
        for item in column_names:
            if item in target_features :
                if df[item].dtype in ['int64', 'float64'] :
                    num_features.append(item)
                elif df[item].dtype in ['object']:
                    print("the used feature is categorical within the VFL linear regression.('not handled yet')")
                    raise ValueError(f"Feature '{item}' is categorical. VFL linear regression currently supports only numerical features.")
    
    preprocessor = ColumnTransformer([('num', StandardScaler(), num_features)])
    # Save the features processor for the future tests and evaluations 
    preprocessor_path = f"../results/ml_{approche}/"  
    os.makedirs(preprocessor_path, exist_ok=True)
    X = preprocessor.fit_transform(df)
    with open(f"../results/ml_{approche}/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
        
    return torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32)



def preprocess_node_data_NN(csv_path,target_features,approche):
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
    preprocessor_path = f"../results/dl_{approche}/"  
    os.makedirs(preprocessor_path, exist_ok=True)
    X = preprocessor.fit_transform(df)
    with open(f"../results/dl_{approche}/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
        #X = preprocessor.transform(df)

    return torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32)




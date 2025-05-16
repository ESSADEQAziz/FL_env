import torch
import torch.nn as nn
import pandas as pd
import os
import json
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
    
def preprocess_features(csv_path,target_features,node_id,approche):
    df = pd.read_csv(csv_path)
    column_names = df.columns

    num_features=[]
    cat_features=[]

    used_features = []

    for item in column_names:
        if item in target_features :
            if df[item].dtype in ['int64', 'float64'] :
                num_features.append(item)
                used_features.append(item)
            elif df[item].dtype in ['object']:
                cat_features.append(item)
                used_features(item)

    # Handle missing values in numerical features
    for col in num_features:
        # Replace NaN values with column mean
        if df[col].isna().any():
            col_mean = df[col].mean()
            df[col].fillna(col_mean, inplace=True)
            # filled the numerical feature col with the mean col_mean

            
    # Handle missing values in categorical features
    for col in cat_features:
        if df[col].isna().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
            # filled the categorical feature col with the mode df[col].mode()[0]

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
    with open(f"{preprocessor_path}preprocessor_{node_id}.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    return torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32),used_features

def save_encoder(self,NODE_ID, model_type="dl_r"):
    """Save the encoder model and metadata for future predictions"""

        
        # Map model type to server directory
    model_type_mapping = {
            "dl_r": "dl_regression",
            "dl_c": "dl_classification",
            "ml_r": "ml_regression",
            "ml_c": "ml_classification"
        }
    

    temp = model_type_mapping.get(model_type, "dl_regression")
    save_dir = f"../results/{temp}/encoders"
        
        # Create directories
    os.makedirs(save_dir, exist_ok=True)
        
        
        # 2. Save features and metadata
    metadata = {
            "features": self.encoder.features,
            "embedding_size": self.embedding_size,
            "data_shape": list(self.data.shape),
            "model_type": model_type
        }  

        # 3. Also save to server directory if it exists
    
    encoder_path = os.path.join(save_dir, f"client_encoder_{NODE_ID}.pth")
    torch.save(self.encoder, encoder_path)
            
    metadata_path = os.path.join(save_dir, f"encoder_metadata_{NODE_ID}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
            
            

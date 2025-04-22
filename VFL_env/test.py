import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer

def preprocess_client_data(csv_path,target_features):
    df = pd.read_csv(csv_path)
    column_names = df.columns
    print("===============================================================")
    print(f"the columns name : {column_names}")
    print("===============================================================")

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
    print(f"the num_features are :{num_features} / and the cat_features are {cat_features}")
    print("===============================================================")

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    X = preprocessor.fit_transform(df)
    print(f"The value of X is : {X}")

    return torch.tensor(X.toarray() if hasattr(X, "toarray") else X, dtype=torch.float32)

def preprocess_server_target(csv_path,target_feature):
    df = pd.read_csv(csv_path)
    y=df[target_feature]

    if y.dtype in ['int64', 'float64'] :
        y_processed = y

    elif y.dtype in ['object']:
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit(y).transform(y)


    return torch.tensor(y_processed, dtype=torch.long)

target_features=["race","anchor_age","gender"]
target_table = "./data/node2_admissions/data.csv"

res=preprocess_client_data(target_table,target_features)
print(res.shape)
print("===============================================================")
print(res.dtype)
print("===============================================================")
# num_values=res
# num_values = min(100, res.size(0))
# first_100 = res[:num_values].detach().cpu().numpy()
# print(num_values)
# print("===============================================================")
# print(first_100)
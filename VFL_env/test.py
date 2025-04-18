import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder ,LabelEncoder
from sklearn.compose import ColumnTransformer

path = './data/node1/patients.csv'
target = "anchor_age"

def preprocess_server_target(csv_path,target_feature):
    df = pd.read_csv(csv_path)

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_feature])
    y = df[target_feature]

        # Identify numeric and categorical features in the feature set
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()


        # Create preprocessor for features
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])


    # Process features using fit_transform for training data
    # X_processed = preprocessor.fit_transform(X)
    
    # Process target if needed (for classification)
    if y.dtype == 'object':  # If target is categorical
        label_encoder = LabelEncoder()
        y_processed = label_encoder.fit(y).transform(y)
    else:  # If target is numerical
        y_processed = y



    return torch.tensor(y_processed, dtype=torch.float32)


y=preprocess_server_target(path,target)
# print(f"the shape of x :{x.shape[0]}")
print(f"the shape of y :{y.shape[0]}")
print("==============================================================================")







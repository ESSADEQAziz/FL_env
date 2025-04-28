import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

def preprocess_node_data_NN(csv_path,features,target):
    df = pd.read_csv(csv_path)
    column_names = df.columns

    num_features=[]
    cat_features=[]

    if df[target].dtype not in ['int64', 'float64']:
        raise ValueError(f"the target feature is not numerical.")

    for item in column_names:
        if item in features :
            if df[item].dtype in ['int64', 'float64'] :
                num_features.append(item)
            elif df[item].dtype in ['object']:
                cat_features.append(item)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])
    
    # Preprocessor for target
    target_scaler = StandardScaler()
    Y = target_scaler.fit_transform(df[[target]])  # Keep as 2D array (n_samples, 1)

    
    X = preprocessor.fit_transform(df[features])
    
  

    # If X is sparse (because of OneHotEncoder), convert to dense
    if hasattr(X, "toarray"):
        X = X.toarray()

    return X , Y

target_table = "./data/node1/labevents.csv"
missing_rate = 0.2
feature_x = ['valuenum']
feature_y = "ref_range_lower"

x,y=preprocess_node_data_NN(target_table,feature_x,feature_y)
print(x)
print("------------------------------------------")
print(y)
print("------------------------------------------")
print(x.shape)
print("------------------------------------------")
print(x.shape)



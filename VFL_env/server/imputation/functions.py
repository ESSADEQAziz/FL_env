import torch.nn as nn
import torch
import pandas as pd
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder , LabelEncoder
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
    
def preprocess_server_num_target(csv_path,target_feature):
    df = pd.read_csv(csv_path)
    y = df[target_feature]

    if y.dtype in ['int64', 'float64'] :
        return torch.tensor(y, dtype=torch.float32)
           
    elif y.dtype in ['object']:
        print("the target feature is not in ['int64', 'float64'] . ")
        return None
        

def process_server_cat_target():
    pass
    # Process target if needed (for classification)
    # if y.dtype == 'object':  # If target is categorical
    #     label_encoder = LabelEncoder()
    #     y_processed = label_encoder.fit(y).transform(y)
    # else:  # If target is numerical
    #     y_processed = y



# class SimpleLinearRegressor(nn.Module):
#     def __init__(self, input_dim):
#         super(SimpleRegressor, self).__init__()
#         self.linear = nn.Linear(input_dim, 1)  # Linear regression

#     def forward(self, x):
#         return self.linear(x)
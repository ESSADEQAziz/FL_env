import pandas as pd
import torch
import torch.nn as nn
import pickle

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.model(x)
    
with open("./server/results/dl_classification/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("./server/results/dl_classification/label_map.pkl", "rb") as f:
    label_map = pickle.load(f)


# Sample input
sample = {
    "race": "Asian",
    "gender": "Female",
    "anchor_age": 67
}

# Convert to DataFrame
df_sample = pd.DataFrame([sample])
print(df_sample)
print("===================================================")

# Preprocess features
# Apply preprocessing
X_sample = preprocessor.transform(df_sample)
print(X_sample)
print("===================================================")
X_tensor = torch.tensor(X_sample.toarray() if hasattr(X_sample, "toarray") else X_sample, dtype=torch.float32)

input_dim = X_tensor.shape[1]
output_dim = len(label_map)

model = SimpleClassifier(input_dim=input_dim, num_classes=output_dim)
model.load_state_dict(torch.load("./server/results/dl_classification/final_vfl_model.pth", map_location=torch.device("cpu")))
model.eval()

with torch.no_grad():
    output = model(X_tensor)
    pred_class = torch.argmax(output, dim=1).item()
    pred_label = label_map[pred_class]

print(f"Predicted insurance class: {pred_label}")

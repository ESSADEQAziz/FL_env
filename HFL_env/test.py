import pandas as pd

csv_file = "C:/Users/BeeClick/Desktop/patients.csv"
column_name = "subject_id"

df = pd.read_csv(csv_file)

print(len(df[column_name].unique()))

# Horizontal Federated Learning with MIMIC-IV Data

This project implements a horizontal federated learning system using MIMIC-IV healthcare data, simulating a scenario where multiple hospitals collaborate on model training without sharing raw patient data.

## Choise of the tables
the choise of the tables is based on features that are likely have missing data and are critical for decesion making. (patients-icustays-admissions-chartevents-labevents) 

## Project Structure

```
HFL_ENV/
├── docker-compose.yml           # Docker Compose configuration
├── server/                      # Federated learning server code
    ├── imputation/              # Starting specific approche        
    │    ├── statictical.py
    │    ├── machine_learning.py
    │    └── deep_learning.py
    ├── logs/                   
    ├── Dockerfile
    ├── requirements.txt
│   └── server_app.py
├── nodes/                      # Federated learning client code
    ├── imputation/             # Implementation of each approche        
    │    ├── statictical_imputation.py
    │    ├── ml_imputation.py
    │    └── dl_imputation.py
    ├── logs/
    ├── results/                # Metrics after the evaluations (json files)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── node_app.py
└── data/                        # Partitioned data (~2MB per file)
    ├── node1/                   # Random patient data
    │   ├── patients.csv
    │   ├── icustays.csv
    │   ├── admissions.csv
    │   ├── chartevents.csv
    │   └── labevents.csv
    ├── node2/                   # Patients aged 40-60
    ├── node3/                   # Patients with diabetes
    ├── node4/                   # Patients with longer ICU stays
    └── node5/                   # Patients with specific insurance type
```

## Data Preparation

The data has been partitioned into 5 nodes with different characteristics to highlight data heterogeneity:

1. **Node 1**: Random patient samples
2. **Node 2**: Patients aged 40-60
3. **Node 3**: Patients with diabetes diagnosis
4. **Node 4**: Patients with longer-than-median ICU stays
5. **Node 5**: Patients with a specific insurance type

Each site has approximately 2MB of data for each file type to keep the simulation manageable.

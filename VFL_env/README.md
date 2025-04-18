# Vertical Federated Learning with MIMIC-IV Data

This project implements a vertical federated learning system using MIMIC-IV healthcare data, simulating a scenario where multiple hospitals collaborate on model training without sharing raw patient data.

---
## Choise of the tables
the choise of the tables is based on features that are likely have missing data and are critical for decesion making. (patients-icustays-admissions-chartevents-labevents) 


---

## Data Preparation
In this setup, **each node contains a different table** (i.e., a different view of the same patients), and all nodes are aligned by `subject_id`.


1. **Load 10GB of Each CSV File**  
   Each file is loaded using pandas with an approximate limit of 10GB of data to fit into memory. Only relevant columns are read to reduce load time.

2. **Select 1 Sample Per Unique Patient (`subject_id`)**  
   For each table:
   - Rows are sorted by a timestamp column (e.g. `admittime`, `charttime`, or `labtime`) to prioritize early records.
   - Duplicates are dropped, keeping only the **first occurrence** of each patient.
   - This results in one row per patient.

3. **Align Patients Across Tables**  
   - Find the intersection of `subject_id`s across all five tables.
   - Reorder all tables so that the rows correspond to the **same patients in the same order**.
   - This guarantees that row `i` in every CSV refers to the same patient.

4. **Save Processed Tables Separately**  
   - The resulting aligned, per-patient datasets are saved to:
1. **Node 1**: features from patients table.
2. **Node 2**: features from admissions table
3. **Node 3**: features from chartevents table.
4. **Node 4**: features from labevents table.
5. **Node 5**: features from icustays table.

---




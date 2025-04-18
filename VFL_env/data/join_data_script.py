import pandas as pd

# Estimated number of rows for ~1GB per table (adjust as needed depending on column count)
ROWS_LIMIT = 25_000_000  # you can tune this depending on file column width

# --- Load a sample of 25 GB per CSV ---

patients = pd.read_csv("patients.csv")  # small enough, load full
admissions = pd.read_csv("admissions.csv", nrows=ROWS_LIMIT)
icustays = pd.read_csv("icustays.csv", nrows=ROWS_LIMIT)
chartevents = pd.read_csv("chartevents.csv", low_memory=False, nrows=ROWS_LIMIT)
labevents = pd.read_csv("labevents.csv", low_memory=False, nrows=ROWS_LIMIT)

# --- Unique subject_ids from patients ---
unique_subject_ids = patients["subject_id"].unique()

# --- Filter and deduplicate for each subject_id ---

# Patients table (already one row per subject)
patients_filtered = patients[patients["subject_id"].isin(unique_subject_ids)]

# Admissions: first admission per patient
admissions_filtered = admissions.sort_values(by=["subject_id", "admittime"]) \
                                .drop_duplicates(subset="subject_id", keep="first")

# ICU stays: first ICU stay per patient
icustays_filtered = icustays.sort_values(by=["subject_id", "intime"]) \
                            .drop_duplicates(subset="subject_id", keep="first")

# Chartevents: first chart event per patient
chartevents_filtered = chartevents.sort_values(by=["subject_id", "charttime"]) \
                                  .drop_duplicates(subset="subject_id", keep="first")

# Labevents: first lab event per patient
labevents_filtered = labevents.sort_values(by=["subject_id", "charttime"]) \
                              .drop_duplicates(subset="subject_id", keep="first")

# --- Align all on common subject_id ---
common_subjects = set(patients_filtered["subject_id"]) \
                & set(admissions_filtered["subject_id"]) \
                & set(icustays_filtered["subject_id"]) \
                & set(chartevents_filtered["subject_id"]) \
                & set(labevents_filtered["subject_id"])

common_subjects = sorted(list(common_subjects))

# --- Filter again to match common subject order ---
def align_table(df, name):
    df_filtered = df[df["subject_id"].isin(common_subjects)]
    return df_filtered.set_index("subject_id").loc[common_subjects].reset_index()


patients_final = align_table(patients_filtered, "patients")
admissions_final = align_table(admissions_filtered, "admissions")
icustays_final = align_table(icustays_filtered, "icustays")
chartevents_final = align_table(chartevents_filtered, "chartevents")
labevents_final = align_table(labevents_filtered, "labevents")

# --- Save results ---

patients_final.to_csv("aligned_patients.csv", index=False)
admissions_final.to_csv("aligned_admissions.csv", index=False)
icustays_final.to_csv("aligned_icustays.csv", index=False)
chartevents_final.to_csv("aligned_chartevents.csv", index=False)
labevents_final.to_csv("aligned_labevents.csv", index=False)



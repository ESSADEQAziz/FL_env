def align_mimic_tables(output_dir="./", rows_limit=20_000_000):
    """
    Aligns and prepares data from multiple MIMIC database tables.
    
    Args:
        output_dir: Directory to save aligned CSV files
        rows_limit: Maximum number of rows to load for large tables
        
    Returns:
        Dict of DataFrames containing the aligned tables
    """
    import pandas as pd
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading MIMIC tables...")
    
    # Load tables with appropriate row limits
    patients = pd.read_csv("patients.csv")  # small enough, load full
    admissions = pd.read_csv("admissions.csv", nrows=rows_limit)
    icustays = pd.read_csv("icustays.csv", nrows=rows_limit)
    chartevents = pd.read_csv("chartevents.csv", low_memory=False, nrows=rows_limit)
    labevents = pd.read_csv("labevents.csv", low_memory=False, nrows=rows_limit)
    
    # Load additional tables
    extracted_lab_results = pd.read_csv("extracted_lab_results.csv", nrows=rows_limit)
    extracted_vital_signs = pd.read_csv("extracted_vital_signs.csv", nrows=rows_limit)
    
    # Get unique subject_ids from patients
    unique_subject_ids = patients["subject_id"].unique()
    print(f"Found {len(unique_subject_ids)} unique patients")
    
    # Filter and deduplicate for each subject_id
    print("Filtering tables to first entry per patient...")
    
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
    
    # Lab results: first lab result per patient
    extracted_lab_filtered = extracted_lab_results.sort_values(by=["subject_id", "charttime"]) \
                                                 .drop_duplicates(subset="subject_id", keep="first")
    
    # Vital signs: first vital signs record per patient
    extracted_vitals_filtered = extracted_vital_signs.sort_values(by=["subject_id", "charttime"]) \
                                                    .drop_duplicates(subset="subject_id", keep="first")
    
    # Find common subjects across all tables
    print("Finding common subjects across all tables...")
    common_subjects = set(patients_filtered["subject_id"]) \
                    & set(admissions_filtered["subject_id"]) \
                    & set(icustays_filtered["subject_id"]) \
                    & set(chartevents_filtered["subject_id"]) \
                    & set(labevents_filtered["subject_id"]) \
                    & set(extracted_lab_filtered["subject_id"]) \
                    & set(extracted_vitals_filtered["subject_id"])
    
    common_subjects = sorted(list(common_subjects))
    print(f"Found {len(common_subjects)} subjects common to all tables")
    
    # Helper function to align table by common subjects
    def align_table(df, name):
        print(f"Aligning {name}...")
        df_filtered = df[df["subject_id"].isin(common_subjects)]
        return df_filtered.set_index("subject_id").loc[common_subjects].reset_index()
    
    # Align all tables to common subjects
    patients_final = align_table(patients_filtered, "patients")
    admissions_final = align_table(admissions_filtered, "admissions")
    icustays_final = align_table(icustays_filtered, "icustays")
    chartevents_final = align_table(chartevents_filtered, "chartevents")
    labevents_final = align_table(labevents_filtered, "labevents")
    extracted_lab_final = align_table(extracted_lab_filtered, "extracted_lab_results")
    extracted_vitals_final = align_table(extracted_vitals_filtered, "extracted_vital_signs")
    
    # Save results
    print(f"Saving aligned tables to {output_dir}...")
    patients_final.to_csv(f"{output_dir}/aligned_patients.csv", index=False)
    admissions_final.to_csv(f"{output_dir}/aligned_admissions.csv", index=False)
    icustays_final.to_csv(f"{output_dir}/aligned_icustays.csv", index=False)
    chartevents_final.to_csv(f"{output_dir}/aligned_chartevents.csv", index=False)
    labevents_final.to_csv(f"{output_dir}/aligned_labevents.csv", index=False)
    extracted_lab_final.to_csv(f"{output_dir}/aligned_lab_results.csv", index=False)
    extracted_vitals_final.to_csv(f"{output_dir}/aligned_vital_signs.csv", index=False)
    
    print("Alignment complete.")
    
    # Return all aligned dataframes in a dict
    return {
        "patients": patients_final,
        "admissions": admissions_final,
        "icustays": icustays_final,
        "chartevents": chartevents_final,
        "labevents": labevents_final,
        "lab_results": extracted_lab_final,
        "vital_signs": extracted_vitals_final
    }

align_mimic_tables()
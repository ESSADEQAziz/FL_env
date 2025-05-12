import pandas as pd


def check_data_structure(node_path):
    """
    Check the structure of the MIMIC-IV data
    """
    # Load the necessary tables
    chart_events = pd.read_csv(f'{node_path}/chartevents.csv')
    patients = pd.read_csv(f'{node_path}/patients.csv')
    
    print("Chart Events Info:")
    print(chart_events.info())
    print("\nChart Events Columns:")
    print(chart_events.columns.tolist())
    print("\nChart Events Sample:")
    print(chart_events.head())
    
    print("\nPatients Info:")
    print(patients.info())
    print("\nPatients Columns:")
    print(patients.columns.tolist())
    print("\nPatients Sample:")
    print(patients.head())

def map_mimic_features():
    """
    Map MIMIC-IV features to their categories and relationships
    """
    feature_mappings = {
        'vital_signs': {
            'chartevents': {
                'heart_rate': ['220045','220046','220047'],  # Heart Rate
                'blood_pressure_systolic': ['220050','220056','220058'],  # Blood Pressure
                'blood_pressure_diastolic': ['220051','220052'],  # Blood Pressure
                'respiratory_rate': ['220210'],  # Respiratory Rate
                'spo2': ['220277']  # SpO2
            }
        },
        'lab_results': {
            'labevents': {
                'blood_glucose': ['50931'],  # Glucose
                'hemoglobin': ['51222','51223','51224','51225','51226','51227'],  # Hemoglobin
                'wbc': ['51221'],  # White Blood Cells
                'platelet_count': ['51264','51265','51266'],  # Platelet Count
                'creatinine': ['50910','50911','50912']  # Creatinine
            }
        }
    }
    return feature_mappings

def prepare_mimic_data(node_path, feature_type='vital_signs'):
    """
    Prepare MIMIC-IV data for missing value analysis
    
    Args:
        node_path: Path to node data directory
        feature_type: 'vital_signs' or 'lab_results'
    """
    # Load the necessary tables
    print("\nLoading data tables...")
    try:
        if feature_type == 'vital_signs':
            events = pd.read_csv(f'{node_path}/chartevents.csv')
            print("Loaded chartevents.csv")
        else:  # lab_results
            events = pd.read_csv(f'{node_path}/labevents.csv')
            print("Loaded labevents.csv")
            
        patients = pd.read_csv(f'{node_path}/patients.csv')
        print("Loaded patients.csv")
        admissions = pd.read_csv(f'{node_path}/admissions.csv')
        print("Loaded admissions.csv")
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), [], []

    print("\n=== Events Table Info ===")
    print("\nColumns in events table:")
    print(events.columns.tolist())
    print("\nFirst few rows of events:")
    print(events.head())
    print("\nData types of events columns:")
    print(events.dtypes)
    print("\nNumber of unique values in each column:")
    for col in events.columns:
        print(f"{col}: {events[col].nunique()} unique values")
    
    # Get feature mappings
    feature_mappings = map_mimic_features()
    
    # Create a list to store DataFrames for each feature
    feature_dfs = []
    
    if feature_type == 'vital_signs':
        print("\nProcessing vital signs...")
        feature_dict = feature_mappings['vital_signs']['chartevents']
        value_col = 'valuenum'
    else:  # lab_results
        print("\nProcessing lab results...")
        feature_dict = feature_mappings['lab_results']['labevents']
        value_col = 'valuenum'
    
    for feature, itemids in feature_dict.items():
        print(f"\nProcessing {feature} with itemids {itemids}")
        
        # Convert itemids to integers
        try:
            itemids = [int(x) for x in itemids]
        except ValueError as e:
            print(f"Error converting itemids to int: {e}")
            continue
            
        # Filter events for the specific itemid
        try:
            feature_data = events[events['itemid'].isin(itemids)].copy()
            print(f"Found {len(feature_data)} rows for {feature}")
        except Exception as e:
            print(f"Error filtering data: {e}")
            continue
        
        if len(feature_data) > 0:
            print(f"Sample of data for {feature}:")
            print(feature_data.head())
            
            # Check if required columns exist
            required_cols = ['subject_id', 'hadm_id', 'charttime' if feature_type == 'vital_signs' else 'charttime', value_col]
            missing_cols = [col for col in required_cols if col not in feature_data.columns]
            if missing_cols:
                print(f"WARNING: Missing required columns: {missing_cols}")
                print(f"Available columns: {feature_data.columns.tolist()}")
                continue
            
            try:
                # Remove rows with NULL values in key columns
                feature_data = feature_data.dropna(subset=['subject_id', 'hadm_id', value_col])
                print(f"After removing NULL values: {len(feature_data)} rows")
                
                # Convert subject_id and hadm_id to appropriate types
                feature_data['subject_id'] = feature_data['subject_id'].astype('Int64')  # Using Int64 to handle NaN
                feature_data['hadm_id'] = feature_data['hadm_id'].astype('Int64')  # Using Int64 to handle NaN
                
                # Convert time to datetime
                time_col = 'charttime' if feature_type == 'vital_signs' else 'charttime'
                feature_data[time_col] = pd.to_datetime(feature_data[time_col])
                
                # Handle the value column
                if feature_type == 'lab_results':
                    # For lab results, convert to float to handle potential decimal values
                    feature_data[value_col] = pd.to_numeric(feature_data[value_col], errors='coerce')
                else:
                    # For vital signs, keep as is
                    feature_data[value_col] = pd.to_numeric(feature_data[value_col], errors='coerce')
                
                # Remove rows where the value is NULL after conversion
                feature_data = feature_data.dropna(subset=[value_col])
                print(f"After cleaning values: {len(feature_data)} rows")
                
                # Rename the value column to the feature name
                feature_data = feature_data.rename(columns={value_col: feature})
                
                # Select only necessary columns
                feature_data = feature_data[['subject_id', 'hadm_id', 'charttime' if feature_type == 'vital_signs' else 'charttime', feature]]
                
                if len(feature_data) > 0:
                    # Append to list only if we have data
                    feature_dfs.append(feature_data)
                    print(f"Added {feature} data to feature_dfs")
                else:
                    print(f"No valid data remaining for {feature}")
                    
            except Exception as e:
                print(f"Error processing {feature} data: {e}")
                continue
    
    if not feature_dfs:
        print("\nNo data found for any features!")
        return pd.DataFrame(), [], []
        
    # Merge all feature DataFrames
    print("\nMerging feature DataFrames...")
    try:
        data = feature_dfs[0]
        for df in feature_dfs[1:]:
            merge_cols = ['subject_id', 'hadm_id', 'charttime' if feature_type == 'vital_signs' else 'charttime']
            data = pd.merge(
                data,
                df,
                on=merge_cols,
                how='outer'
            )
    except Exception as e:
        print(f"Error merging feature DataFrames: {e}")
        return pd.DataFrame(), [], []
    
    print("\nMerging with patient demographics...")
    try:
        # Merge with patient demographics
        data = pd.merge(
            data,
            patients[['subject_id', 'gender', 'anchor_age']],
            on='subject_id',
            how='left'
        )
    except Exception as e:
        print(f"Error merging with demographics: {e}")
        return pd.DataFrame(), [], []
    
    # Select features for missingness and imputation
    if feature_type == 'vital_signs':
        features = list(feature_mappings['vital_signs']['chartevents'].keys())
    else:  # lab_results
        features = list(feature_mappings['lab_results']['labevents'].keys())
    
    # Select features for imputation
    imputation_features = features + ['gender', 'anchor_age']
    
    try:
        # Sort by subject_id, hadm_id, and time
        sort_cols = ['subject_id', 'hadm_id']
        sort_cols.append('charttime' if feature_type == 'vital_signs' else 'charttime')
        data = data.sort_values(sort_cols)
        
        # Reset index
        data = data.reset_index(drop=True)
        
        print("\nFinal data shape:", data.shape)
        print("\nFinal data columns:", data.columns.tolist())
        print("\nSample of final data:")
        print(data.head())
        
        return data, features, imputation_features
    except Exception as e:
        print(f"Error in final data processing: {e}")
        return pd.DataFrame(), [], []

if __name__ == "__main__":

    for i in range(5) :
        x,_,_ = prepare_mimic_data(f'../data/node{i+1}',"vital_signs")
        x.to_csv(f'extracted_vital_signs_node{i+1}.csv', index=False)

        y,_,_ = prepare_mimic_data(f'../data/node{i+1}',"lab_results")
        y.to_csv(f'extracted_lab_results_node{i+1}.csv', index=False)



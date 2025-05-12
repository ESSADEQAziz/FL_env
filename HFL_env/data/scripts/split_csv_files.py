import pandas as pd
import os
import math

def create_folders():
    # Create 5 folders if they don't exist
    for i in range(1, 6):
        folder_name = f'part{i}'
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

def split_csv_file(file_path):
    print(f"Processing {file_path}...")
    
    # Get total number of rows without loading entire file
    total_rows = sum(1 for _ in open(file_path, 'r'))
    chunk_size = math.ceil(total_rows / 5)
    
    # Process file in chunks
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        if i >= 5:  # Only create 5 parts
            break
            
        output_path = f'part{i+1}/{os.path.basename(file_path)}'
        chunk.to_csv(output_path, index=False)
        print(f"Created {output_path}")

if __name__ == "__main__":

    # Create the folders
    create_folders()
    
    # Get all CSV files in directory path
    directory_path = '.'
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    # Process each CSV file
    for csv_file in csv_files:
        try:
            split_csv_file(csv_file)
        except Exception as e:
            print(f"Error processing {csv_file}: {str(e)}")

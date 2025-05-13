import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from IPython.display import display
import torch
import torch.nn as nn
import pickle
import json
import os
from scipy import stats
import warnings

def analyze_feature(df, feature):
    """
    Analyze a single feature in a dataframe, providing comprehensive statistics and visualizations.
    
    Args:
        df: pandas DataFrame
        feature: name of the column to analyze
    """
    # Check if feature exists
    if feature not in df.columns:
        print(f"Error: Feature '{feature}' not found in DataFrame columns.")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Get the series
    series = df[feature]
    
    print(f"\n{'='*50}")
    print(f"ANALYSIS OF FEATURE: {feature}")
    print(f"{'='*50}")
    
    # Basic information
    print("\n--- Basic Information ---")
    print(f"Data type: {series.dtype}")
    null_count = series.isna().sum()
    null_percent = (null_count / len(series)) * 100
    print(f"Missing values: {null_count} ({null_percent:.2f}%)")
    print(f"Unique values: {series.nunique()} ({series.nunique()/len(series)*100:.2f}% of total)")
    
    # For categorical features
    if series.dtype == 'object' or series.dtype.name == 'category' or series.nunique() < 10:
        print("\n--- Categorical Analysis ---")
        value_counts = series.value_counts()
        value_percent = series.value_counts(normalize=True) * 100
        
        # Create a DataFrame with counts and percentages
        cat_df = pd.DataFrame({
            'Count': value_counts,
            'Percentage': value_percent
        })
        print(cat_df)
        
        # Plot bar chart for categorical data
        plt.figure(figsize=(10, 6))
        sns.countplot(y=feature, data=df, order=value_counts.index)
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.show()
    
    # For numeric features
    if np.issubdtype(series.dtype, np.number):
        print("\n--- Numeric Analysis ---")
        
        # Basic statistics
        stats_df = pd.DataFrame({
            'Statistic': ['count', 'mean', 'std', 'min', '25%', '50% (median)', '75%', 'max', 
                         'skewness', 'kurtosis', 'IQR', 'range'],
            'Value': [
                series.count(),
                series.mean(),
                series.std(),
                series.min(),
                series.quantile(0.25),
                series.median(),
                series.quantile(0.75),
                series.max(),
                series.skew(),
                series.kurtosis(),
                series.quantile(0.75) - series.quantile(0.25),
                series.max() - series.min()
            ]
        })
        print(stats_df)
        
        # Check for outliers using IQR method
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        
        print(f"\nOutliers (IQR method): {len(outliers)} ({len(outliers)/len(series)*100:.2f}% of non-null values)")
        print(f"Outlier bounds: Lower < {lower_bound:.2f} or Upper > {upper_bound:.2f}")
        
        if len(outliers) > 0 and len(outliers) < 20:
            print("Outlier values:")
            print(outliers.values)
        
        # Normality test (Shapiro-Wilk)
        if series.count() > 3 and series.count() <= 5000:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    stat, p_value = stats.shapiro(series.dropna())
                    print(f"\nShapiro-Wilk Normality Test: statistic={stat:.4f}, p-value={p_value:.4f}")
                    alpha = 0.05
                    if p_value > alpha:
                        print(f"Feature appears normally distributed (p > {alpha})")
                    else:
                        print(f"Feature does not appear normally distributed (p < {alpha})")
            except:
                print("Couldn't perform Shapiro-Wilk normality test")
        elif series.count() > 5000:
            print("\nSkipping Shapiro-Wilk normality test due to large sample size (n > 5000).")
            
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Histogram
        plt.subplot(2, 2, 1)
        sns.histplot(series, kde=True)
        plt.title(f'Distribution of {feature}')
        
        # Box plot
        plt.subplot(2, 2, 2)
        sns.boxplot(y=series)
        plt.title(f'Box Plot of {feature}')
        
        # Q-Q plot
        plt.subplot(2, 2, 3)
        stats.probplot(series.dropna(), dist="norm", plot=plt)
        plt.title(f'Q-Q Plot of {feature}')
        
        # KDE plot
        plt.subplot(2, 2, 4)
        sns.kdeplot(series)
        plt.title(f'KDE Plot of {feature}')
        
        plt.tight_layout()
        plt.show()
    
    # Time series analysis if timestamp
    if pd.api.types.is_datetime64_any_dtype(series) or feature.lower() in ['date', 'time', 'timestamp', 'datetime']:
        print("\n--- Time Series Analysis ---")
        
        try:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(series):
                series = pd.to_datetime(series, errors='coerce')
                
            print(f"Date range: {series.min()} to {series.max()}")
            print(f"Time span: {series.max() - series.min()}")
            
            # Count by year, month, and day of week
            if (series.max() - series.min()).days > 365:
                yearly_counts = series.dt.year.value_counts().sort_index()
                print("\nCounts by year:")
                print(yearly_counts)
            
            monthly_counts = series.dt.month.value_counts().sort_index()
            print("\nCounts by month:")
            print(monthly_counts)
            
            dow_counts = series.dt.day_name().value_counts()
            print("\nCounts by day of week:")
            print(dow_counts)
            
            # Plot time series
            plt.figure(figsize=(12, 8))
            
            # By month
            plt.subplot(2, 1, 1)
            sns.countplot(x=series.dt.month, order=range(1, 13))
            plt.title(f'Distribution of {feature} by Month')
            plt.xlabel('Month')
            
            # By day of week
            plt.subplot(2, 1, 2)
            sns.countplot(y=series.dt.day_name(), order=['Monday', 'Tuesday', 'Wednesday', 
                                                       'Thursday', 'Friday', 'Saturday', 'Sunday'])
            plt.title(f'Distribution of {feature} by Day of Week')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Couldn't perform time series analysis: {e}")
    
    # Correlation with other numeric features
    if np.issubdtype(series.dtype, np.number) and len(df.select_dtypes(include=[np.number]).columns) > 1:
        print("\n--- Correlation with Other Numeric Features ---")
        
        # Calculate correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = df[numeric_cols].corr()[feature].sort_values(ascending=False)
        
        # Remove self-correlation
        correlations = correlations[correlations.index != feature]
        
        corr_df = pd.DataFrame({
            'Feature': correlations.index,
            'Pearson Correlation': correlations.values
        })
        print(corr_df)
        
        # Plot top and bottom correlations
        if len(correlations) > 0:
            plt.figure(figsize=(10, 6))
            
            # Get top 10 and bottom 10 correlations
            top_corr = correlations.abs().sort_values(ascending=False).head(10).index
            
            # Create correlation plot for selected features
            if len(top_corr) > 0:
                corr_data = df[[feature] + list(top_corr)].corr()
                mask = np.triu(np.ones_like(corr_data, dtype=bool))
                sns.heatmap(corr_data, mask=mask, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
                plt.title(f'Correlation of {feature} with Other Features')
                plt.tight_layout()
                plt.show()
                
                # Scatter plots for the top 3 correlations
                if len(top_corr) >= 3:
                    plt.figure(figsize=(15, 5))
                    for i, col in enumerate(top_corr[:3]):
                        plt.subplot(1, 3, i+1)
                        sns.scatterplot(x=col, y=feature, data=df)
                        plt.title(f'{feature} vs {col}')
                    plt.tight_layout()
                    plt.show()
                
    # Return the analysis as a dictionary
    analysis_summary = {
        'feature': feature,
        'dtype': str(series.dtype),
        'missing_count': null_count,
        'missing_percent': null_percent,
        'unique_values': series.nunique(),
    }
    
    if np.issubdtype(series.dtype, np.number):
        analysis_summary.update({
            'mean': series.mean(),
            'median': series.median(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'skewness': series.skew(),
            'kurtosis': series.kurtosis(),
            'has_outliers': len(outliers) > 0,
            'outliers_count': len(outliers)
        })
    
    print("\nAnalysis complete!")

def analyze_dataframe(df=None, csv_path=None):
    """
    Display basic information about a dataframe with simple table formatting.
    
    Args:
        df: pandas DataFrame (optional if csv_path is provided)
        csv_path: Path to CSV file (optional if df is provided)
    
    Returns:
        The dataframe being analyzed
    """
    # Load data if dataframe not provided
    if df is None:
        if csv_path is None:
            print("Error: Either df or csv_path must be provided.")
            return None
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Dataset loaded from: {csv_path}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
    
    # Basic dataframe information
    print("\n## DATAFRAME OVERVIEW")
    print("-" * 40)
    
    overview_data = {
        "Metric": ["Rows", "Columns", "Total Elements", "Memory Usage (MB)"],
        "Value": [
            df.shape[0], 
            df.shape[1], 
            df.size, 
            f"{df.memory_usage(deep=True).sum() / (1024 * 1024):.2f}"
        ]
    }
    overview_df = pd.DataFrame(overview_data)
    display(overview_df)
    
    # Column information with missingness stats
    print("\n## COLUMN INFORMATION")
    print("-" * 40)
    
    column_info = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_rate = missing_count / len(df)
        
        column_info.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Unique Values": df[col].nunique(),
            "Unique %": f"{(df[col].nunique() / len(df)) * 100:.1f}%",
            "Missing Count": missing_count,
            "Missing %": f"{missing_rate:.2%}"
        })
    
    info_df = pd.DataFrame(column_info)
    display(info_df)
    
    # Sample data
    print("\n## SAMPLE DATA (First 5 rows)")
    print("-" * 40)
    display(df.head())
    
    # Missingness summary
    print("\n## MISSINGNESS SUMMARY")
    print("-" * 40)
    
    # Count features with missing values
    features_with_missing = info_df[info_df["Missing Count"] > 0]["Column"].tolist()
    if features_with_missing:
        print(f"Features with missing values: {len(features_with_missing)} out of {len(df.columns)}")
        print(f"Features with missing values: {', '.join(features_with_missing)}")
        
        # Sort by missing percentage in descending order
        missing_sorted = info_df.sort_values("Missing Count", ascending=False)
        print("\nTop features with highest missingness:")
        
        top_missing = missing_sorted[["Column", "Missing Count", "Missing %"]].head(10)
        display(top_missing)
    else:
        print("No missing values found in the dataset.")
    
    return df

def calculate_spearman_correlation(df, target_col=None):
    """
    Calculate and print Spearman rank correlation for all numeric columns or against a specific target,
    with proper handling of NaN values.
    
    Args:
        df: DataFrame containing the data
        target_col: Optional target column to correlate against
    """
    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Check for and report NaN values
    total_rows = len(numeric_df)
    nan_counts = numeric_df.isna().sum()
    columns_with_nans = nan_counts[nan_counts > 0]
    
    if len(columns_with_nans) > 0:
        print("\n=== NaN Values in Numeric Columns ===")
        for col, count in columns_with_nans.items():
            print(f"  - {col}: {count} NaN values ({count/total_rows:.2%})")
        
        print("\nNote: NaN values will be automatically excluded from pairwise correlation calculations.")
    
    if target_col:
        # Correlation against specific target
        if target_col not in numeric_df.columns:
            print(f"Error: {target_col} is not a numeric column in the dataframe")
            return
        
        # Check NaNs in target column
        target_nans = numeric_df[target_col].isna().sum()
        if target_nans > 0:
            print(f"\nWarning: Target column '{target_col}' has {target_nans} NaN values ({target_nans/total_rows:.2%}).")
            print("These rows will be excluded when calculating correlations.")
        
        # Calculate correlations (pandas automatically handles NaNs with pairwise deletion)
        corr = numeric_df.corrwith(numeric_df[target_col], method='spearman').sort_values(ascending=False)
        
        # Drop NaN correlation results
        if corr.isna().any():
            print("\nWarning: Some features have NaN correlation with the target.")
            print("This typically happens when a feature has no valid pairs with the target or contains constant values.")
            print("These features will be removed from the results:")
            for col in corr[corr.isna()].index:
                print(f"  - {col}")
            
            corr = corr.dropna()
        
        # Create DataFrame with results
        corr_df = pd.DataFrame(corr, columns=['Spearman Correlation'])
        
        # Report number of observations used
        valid_pairs = {}
        for col in corr_df.index:
            if col != target_col:
                valid_count = numeric_df[[col, target_col]].dropna().shape[0]
                valid_pairs[col] = valid_count
        
        corr_df['Valid Observations'] = pd.Series(valid_pairs)
        corr_df['% of Data Used'] = (corr_df['Valid Observations'] / total_rows * 100).round(2).astype(str) + '%'
        
        print("\n=== Spearman Rank Correlation with target:", target_col, "===")
        print(corr_df)
        
        # Plot the correlation values as a bar chart (only for correlations, not metadata)
        plt.figure(figsize=(10, 6))
        plot_data = corr_df.copy()
        # Sort by absolute correlation value for better visualization
        plot_data = plot_data.sort_values('Spearman Correlation', key=abs, ascending=False)
        
        sns.barplot(x=plot_data.index, y='Spearman Correlation', data=plot_data)
        plt.xticks(rotation=90)
        plt.title(f'Spearman Correlation with {target_col}')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)  # Add a horizontal line at y=0
        plt.tight_layout()
        plt.show()
        
        return corr_df
    
    else:
        # Full correlation matrix (pandas handles NaNs with pairwise deletion)
        corr = numeric_df.corr(method='spearman')
        
        # Create a counts matrix to show number of observations used for each pair
        obs_counts = pd.DataFrame(index=corr.index, columns=corr.columns)
        for i in corr.index:
            for j in corr.columns:
                obs_counts.loc[i, j] = numeric_df[[i, j]].dropna().shape[0]
        
        print("\n=== Spearman Rank Correlation Matrix ===")
        print(corr)
        
        print("\n=== Number of observations used for each correlation pair ===")
        print(obs_counts)
        
        # Calculate percentage of data used
        pct_used = (obs_counts / total_rows * 100).round(2)
        print("\n=== Percentage of data used for each correlation pair ===")
        print(pct_used)
        
        # Check for any low data usage pairs
        low_data_pairs = []
        threshold = 0.7  # Flag pairs using less than 70% of the data
        for i in pct_used.index:
            for j in pct_used.columns:
                if i != j and pct_used.loc[i, j] < threshold * 100:
                    low_data_pairs.append((i, j, pct_used.loc[i, j]))
        
        if low_data_pairs:
            print("\nWarning: Some correlation pairs use a low percentage of the data:")
            for i, j, pct in sorted(low_data_pairs, key=lambda x: x[2]):
                print(f"  - {i} vs {j}: {pct:.1f}% of data used")
        
        # Plot heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, 
                    center=0, square=True, linewidths=.5, annot=True, fmt=".2f")
        plt.title("Spearman Correlation Matrix")
        plt.tight_layout()
        plt.show()
        
        return corr

def calculate_mutual_information(df, target_col):
    """
    Calculate and print mutual information between features and target,
    with pairwise deletion of NaN values for each feature-target pair.
    
    Args:
        df: DataFrame containing the data
        target_col: Target column to calculate MI against
    """
    # Get numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if target_col not in numeric_df.columns:
        print(f"Error: {target_col} is not a numeric column in the dataframe")
        return
    
    # Check for and report NaN values
    total_rows = len(numeric_df)
    nan_counts = numeric_df.isna().sum()
    columns_with_nans = nan_counts[nan_counts > 0]
    
    if len(columns_with_nans) > 0:
        print("\n=== NaN Values in Numeric Columns ===")
        for col, count in columns_with_nans.items():
            print(f"  - {col}: {count} NaN values ({count/total_rows:.2%})")
        
        print("\nNote: NaN values will be excluded on a pairwise basis for each feature-target pair.")
    
    # Check NaNs in target column
    target_nans = numeric_df[target_col].isna().sum()
    if target_nans > 0:
        print(f"\nWarning: Target column '{target_col}' has {target_nans} NaN values ({target_nans/total_rows:.2%}).")
        print("Rows with NaN in the target will be excluded from all calculations.")
    
    # Get features excluding target
    features = [col for col in numeric_df.columns if col != target_col]
    
    # Calculate MI for each feature with pairwise deletion of NaNs
    mi_scores = []
    valid_pairs = {}
    
    for feature in features:
        # Create a clean subset with this feature and the target
        pair_df = numeric_df[[feature, target_col]].dropna()
        valid_count = len(pair_df)
        valid_pairs[feature] = valid_count
        
        # Skip features with too few valid pairs
        if valid_count < 5:
            print(f"Warning: Feature '{feature}' has only {valid_count} valid pairs with target. Skipping MI calculation.")
            mi_scores.append((feature, np.nan))
            continue
        
        # Calculate MI for this pair
        try:
            X = pair_df[[feature]].values  # sklearn expects 2D array
            y = pair_df[target_col].values
            mi = mutual_info_regression(X, y)[0]  # [0] because we get a 1-element array
            mi_scores.append((feature, mi))
        except Exception as e:
            print(f"Error calculating MI for feature '{feature}': {e}")
            mi_scores.append((feature, np.nan))
    
    # Create DataFrame with scores
    mi_df = pd.DataFrame(mi_scores, columns=['Feature', 'MI Score'])
    
    # Add information about valid observations
    valid_obs_series = pd.Series(valid_pairs)
    # Make sure the indices match
    mi_df['Valid Observations'] = mi_df['Feature'].map(valid_obs_series)
    mi_df['% of Data Used'] = (mi_df['Valid Observations'] / total_rows * 100).round(2).astype(str) + '%'
    
    # Sort by MI Score, handling NaN values
    mi_df = mi_df.sort_values('MI Score', ascending=False, na_position='last')
    
    # Handle cases where some features have NaN MI scores
    if mi_df['MI Score'].isna().any():
        print("\nWarning: Some features have NaN Mutual Information scores with the target.")
        print("This typically happens when a feature has too few valid pairs with the target.")
        print("These features will be shown at the end of the results.")
    
    print("\n=== Mutual Information with target:", target_col, "===")
    print(mi_df)
    
    # Plot MI scores (excluding NaN values)
    if len(mi_df.dropna(subset=['MI Score'])) > 0:
        plt.figure(figsize=(10, 6))
        plot_data = mi_df.dropna(subset=['MI Score']).copy()
        sns.barplot(x='MI Score', y='Feature', data=plot_data)
        plt.title(f'Mutual Information Scores (target: {target_col})')
        plt.tight_layout()
        plt.show()
    else:
        print("No valid MI scores to plot.")
    
    # Check for any low data usage pairs
    low_data_pairs = mi_df[mi_df['Valid Observations'] < 0.7 * total_rows]
    if len(low_data_pairs) > 0:
        print("\nWarning: Some feature-target pairs use a low percentage of the data:")
        for _, row in low_data_pairs.iterrows():
            print(f"  - {row['Feature']}: {row['% of Data Used']} of data used")
    
    return mi_df

def introduce_missingness(df, feature, missing_rate, pattern='MCAR', target=None, seed=42):
    """
    Introduce missing values in a specific feature of a DataFrame with a given pattern and rate,
    while saving only the original values paired with charttime.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        feature (str): Name of the feature to introduce missing values
        missing_rate (float): Rate of missing values to introduce (between 0 and 1)
        pattern (str): Pattern of missingness ('MCAR', 'MAR', or 'MNAR')
        target (str, optional): Name of the target variable (needed for MAR pattern)
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (df_with_missing, ground_truth) where:
            - df_with_missing: DataFrame with missing values introduced
            - ground_truth: DataFrame with only original values and charttime
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Check if feature exists in DataFrame
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' not found in DataFrame columns: {df.columns.tolist()}")
    
    # Check if charttime exists
    if 'charttime' not in df.columns:
        raise ValueError("Column 'charttime' not found in DataFrame.")
    
    # Check if we need target for MAR pattern
    if pattern == 'MAR' and (target is None or target not in df.columns):
        raise ValueError(f"Target variable is required for MAR pattern. Target '{target}' not found.")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Create a ground truth DataFrame with only original values and charttime
    ground_truth = pd.DataFrame(index=df_copy.index)
    ground_truth['original_values'] = df_copy[feature].copy()
    ground_truth['charttime'] = df_copy['charttime'].copy()
    
    # Check for existing NaN values
    original_mask = df_copy[feature].isna()
    existing_nan_count = original_mask.sum()
    n_samples = len(df_copy)
    existing_nan_rate = existing_nan_count / n_samples
    
    # Calculate how many values to make missing
    if existing_nan_rate > 0:
        print(f"Warning: Feature '{feature}' already has {existing_nan_rate:.2%} missing values.")
        print("Using only rows with existing values to introduce new missingness.")
    
    # Focus only on non-missing values for introducing missingness
    valid_mask = ~df_copy[feature].isna()
    valid_indices = df_copy.index[valid_mask].tolist()
    
    if len(valid_indices) == 0:
        raise ValueError(f"Feature '{feature}' has no valid values. Cannot introduce missingness.")
    
    # Calculate number of values to make missing
    n_to_make_missing = int(n_samples * missing_rate - existing_nan_count)
    
    if n_to_make_missing <= 0:
        print(f"Feature already has {existing_nan_rate:.2%} missing values, "
              f"which is greater than or equal to the requested {missing_rate:.2%}.")
        return df_copy, ground_truth
    
    if n_to_make_missing > len(valid_indices):
        print(f"Warning: Cannot introduce {n_to_make_missing} missing values as only "
              f"{len(valid_indices)} non-missing values are available.")
        n_to_make_missing = len(valid_indices)
    
    # Select indices to make missing based on pattern
    missing_indices = []
    
    if pattern == 'MCAR':
        # Missing Completely At Random
        missing_indices = np.random.choice(valid_indices, n_to_make_missing, replace=False)
    
    elif pattern == 'MAR':
        # Missing At Random - depends on target variable
        # Create a DataFrame with valid indices and target values
        target_df = pd.DataFrame({
            'index': valid_indices,
            'target_value': df_copy.loc[valid_indices, target].values
        })
        
        # Sort by target values and select indices with lowest target values
        target_df = target_df.sort_values('target_value')
        missing_indices = target_df['index'].head(n_to_make_missing).tolist()
    
    elif pattern == 'MNAR':
        # Missing Not At Random - depends on feature value itself
        # Create a DataFrame with valid indices and feature values
        feature_df = pd.DataFrame({
            'index': valid_indices,
            'feature_value': df_copy.loc[valid_indices, feature].values
        })
        
        # Sort by feature values and select indices with lowest feature values
        feature_df = feature_df.sort_values('feature_value')
        missing_indices = feature_df['index'].head(n_to_make_missing).tolist()
    
    else:
        raise ValueError("Pattern must be one of: 'MCAR', 'MAR', 'MNAR'")
    
    # Introduce missing values in the copy
    df_copy.loc[missing_indices, feature] = np.nan
    
    # Calculate final missing rate for verification
    final_missing_count = df_copy[feature].isna().sum()
    final_missing_rate = final_missing_count / n_samples
    
    print(f"Missingness pattern: {pattern}")
    print(f"Original missing rate: {existing_nan_rate:.2%}")
    print(f"Added {n_to_make_missing} missing values")
    print(f"Final missing rate: {final_missing_rate:.2%}")
    
    return df_copy, ground_truth

def prepare_clean_dataset(df, feature1, feature2):
    """
    Create a new DataFrame with only rows that have non-missing values for both specified features.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        feature1 (str): First feature name to check for missing values
        feature2 (str): Second feature name to check for missing values
        
    Returns:
        pandas.DataFrame: Clean dataset with no missing values in both specified features
    """
    # Filter rows with non-missing values in both features
    clean_df = df.dropna(subset=[feature1, feature2])
    
    # Report statistics
    original_count = len(df)
    clean_count = len(clean_df)
    
    print(f"Original dataset: {original_count} rows")
    print(f"Clean dataset: {clean_count} rows ({clean_count/original_count:.2%} of original)")
    print(f"\nMissing values summary:")
    print(f"{feature1}: {df[feature1].isna().sum()} missing values")
    print(f"{feature2}: {df[feature2].isna().sum()} missing values")
    print(f"Rows missing either feature: {original_count - clean_count}")
    
    return clean_df

def load_model_for_prediction(model_path, preprocessor_dir, approach):
    """
    Load the saved model state_dict and all necessary preprocessors.
    
    Args:
        model_path: Path to the saved model (.pth file)
        preprocessor_dir: Directory containing preprocessors
        approach: 'dl_r', 'ml_r', 'dl_c', or 'ml_c'
    
    Returns:
        model: Loaded PyTorch model
        feature_preprocessor: Loaded feature preprocessor
        target_transformer: Target scaler or label map
    """
    # Extract model info from path
    info_path = model_path.replace("model_round", "model_info_round").replace(".pth", ".json")
    
    if not os.path.exists(info_path):
        # Try to infer the model type from the approach
        if approach == "dl_r":
            model_info = {"model_type": "dl_r", "input_dim": None}
        elif approach == "ml_r":
            model_info = {"model_type": "ml_r", "input_dim": None}
        elif approach == "dl_c":
            model_info = {"model_type": "dl_c", "input_dim": None, "num_classes": None}
        elif approach == "ml_c":
            model_info = {"model_type": "ml_c", "input_dim": None, "num_classes": None}
        else:
            raise ValueError(f"Cannot determine model type from approach: {approach}")
        
        print(f"Warning: Model info file not found at {info_path}. Using default values.")
    else:
        # Load model info
        with open(info_path, 'r') as f:
            model_info = json.load(f)
    
    # Load feature preprocessor to determine input dimensions if needed
    with open(os.path.join(preprocessor_dir, "feature_preprocessor.pkl"), "rb") as f:
        feature_preprocessor = pickle.load(f)
    
    # Determine input dimensions from preprocessor if not available in model_info
    if model_info["input_dim"] is None:
        # Check if the preprocessor has a 'named_transformers_' attribute (ColumnTransformer)
        if hasattr(feature_preprocessor, 'named_transformers_'):
            # Get a sample feature vector to determine dimensions
            try:
                # Create a small dummy DataFrame with the right columns
                dummy_data = {}
                for name, transformer, columns in feature_preprocessor.transformers_:
                    for col in columns:
                        if name == 'num':
                            dummy_data[col] = [0.0]
                        else:
                            dummy_data[col] = ["dummy"]
                
                dummy_df = pd.DataFrame(dummy_data)
                transformed = feature_preprocessor.transform(dummy_df)
                
                # Get the shape of the transformed data
                if hasattr(transformed, 'toarray'):
                    transformed = transformed.toarray()
                
                model_info["input_dim"] = transformed.shape[1]
                print(f"Inferred input dimension: {model_info['input_dim']}")
            except Exception as e:
                print(f"Error inferring input dimensions: {e}")
                # Use a default value
                model_info["input_dim"] = 10
                print(f"Using default input dimension: {model_info['input_dim']}")
        else:
            # Use a default value if we can't determine
            model_info["input_dim"] = 10
            print(f"Using default input dimension: {model_info['input_dim']}")
    
    # Determine number of classes for classification models if needed
    if approach in ['dl_c', 'ml_c'] and model_info.get("num_classes") is None:
        # Try to load label map to determine number of classes
        try:
            with open(os.path.join(preprocessor_dir, "label_map.pkl"), "rb") as f:
                label_map = pickle.load(f)
                model_info["num_classes"] = len(label_map)
                print(f"Inferred number of classes: {model_info['num_classes']}")
        except Exception as e:
            print(f"Error inferring number of classes: {e}")
            # Use a default value
            model_info["num_classes"] = 2
            print(f"Using default number of classes: {model_info['num_classes']}")
    
    # Create model instance based on info
    if model_info["model_type"] == "dl_r":
        model = SimpleRegressor(input_dim=model_info["input_dim"])
    elif model_info["model_type"] == "ml_r":
        model = LinearRegressionModel(input_dim=model_info["input_dim"])
    elif model_info["model_type"] == "ml_c":
        model = LogisticRegressionModel(input_dim=model_info["input_dim"], 
                                        output_dim=model_info["num_classes"])
    elif model_info["model_type"] == "dl_c":
        model = SimpleClassifier(input_dim=model_info["input_dim"], 
                                num_classes=model_info["num_classes"])
    else:
        raise ValueError(f"Unknown model type: {model_info['model_type']}")
    
    # Load state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load target transformer based on approach
    if approach in ['dl_r', 'ml_r']:
        # Regression - load target scaler
        with open(os.path.join(preprocessor_dir, "target_scaler.pkl"), "rb") as f:
            target_transformer = pickle.load(f)
    else:
        # Classification - load label map
        with open(os.path.join(preprocessor_dir, "label_map.pkl"), "rb") as f:
            target_transformer = pickle.load(f)
    
    return model, feature_preprocessor, target_transformer

def predict(model, df, features, feature_preprocessor, target_transformer, approach):
    """
    Make predictions using a saved model with proper preprocessing.
    
    Args:
        model: Loaded PyTorch model
        df: DataFrame with input data
        features: List of feature names
        feature_preprocessor: Loaded feature preprocessor
        target_transformer: Target scaler or label map
        approach: 'dl_r', 'ml_r', 'dl_c', or 'ml_c'
    
    Returns:
        predictions: Processed predictions in original scale
    """
    # Preprocess features using saved preprocessor
    X = feature_preprocessor.transform(df[features])
    
    # Convert to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # Convert to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Handle NaN values if any
    if torch.isnan(X_tensor).any():
        print("Warning: NaN values detected in input data. Handling NaNs...")
        # Get column information from preprocessor to determine numerical vs categorical
        num_features = []
        cat_features = []
        # Extract information from the ColumnTransformer
        for name, transformer, columns in feature_preprocessor.transformers_:
            if name == 'num':
                num_features_count = len(columns)
                num_idx = (0, num_features_count)
            elif name == 'cat':
                cat_features_count = X.shape[1] - num_features_count
                cat_idx = (num_features_count, X.shape[1])
        
        # Handle NaNs by feature type
        feature_types = {}
        if 'num_idx' in locals():
            feature_types[num_idx] = 'numerical'
        if 'cat_idx' in locals():
            feature_types[cat_idx] = 'categorical'
        
        # Apply NaN handling for each feature type
        for range_or_idx, feat_type in feature_types.items():
            if isinstance(range_or_idx, tuple):
                # Range of columns
                start, end = range_or_idx
                X_subset = X_tensor[:, start:end]
                if feat_type == 'numerical':
                    # For numerical features, use column-wise mean imputation
                    column_means = torch.nanmean(X_subset, dim=0, keepdim=True)
                    nan_mask = torch.isnan(X_subset)
                    for col_idx in range(X_subset.shape[1]):
                        col_mask = nan_mask[:, col_idx]
                        if col_mask.any():
                            X_subset[col_mask, col_idx] = column_means[0, col_idx]
                else:
                    # For categorical features, fill with zeros
                    X_subset = torch.nan_to_num(X_subset, nan=0.0)
                
                # Put back into X_tensor
                X_tensor[:, start:end] = X_subset
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor)
    
    # Convert to numpy
    predictions_np = predictions.numpy()
    # Process predictions based on approach
    if approach in ['dl_r', 'ml_r']:
        # Regression - inverse transform to get original scale
        predictions_original = target_transformer.inverse_transform(predictions_np)
        return predictions_original
    else:
        # Classification - convert to class labels
        if predictions_np.shape[1] > 1:
            # Multi-class: get class with highest probability
            pred_classes = np.argmax(predictions_np, axis=1)
            # Map indices to labels
            return [target_transformer[i] for i in pred_classes]
        else:
            # Binary classification
            pred_classes = (predictions_np > 0.5).astype(int)
            # Map 0/1 to actual class labels
            return [target_transformer[i[0]] for i in pred_classes]

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)    

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

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  

def predict_with_model(model, df, features, feature_preprocessor, target_transformer, approach):
    """
    Make predictions using a loaded model with proper preprocessing and NaN handling.
    
    Args:
        model: Loaded PyTorch model
        df: DataFrame with input data
        features: List of feature names
        feature_preprocessor: Loaded feature preprocessor
        target_transformer: Target scaler or label map
        approach: 'dl_r', 'ml_r', 'dl_c', or 'ml_c'
    
    Returns:
        predictions: Processed predictions in original scale
    """
    # Determine feature types based on preprocessor
    feature_types = {}
    categorical_features = []
    numerical_features = []
    
    # Extract feature types from the ColumnTransformer
    if hasattr(feature_preprocessor, 'transformers_'):
        for name, transformer, columns in feature_preprocessor.transformers_:
            if name == 'num':
                numerical_features.extend(columns)
            elif name == 'cat':
                categorical_features.extend(columns)
    
    # Preprocess features using saved preprocessor
    X = feature_preprocessor.transform(df[features])
    
    # Convert to dense if needed
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # Convert to torch tensor
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Identify feature ranges for each type
    if numerical_features and categorical_features:
        # Start index for categorical features depends on where numerical features end
        # This is an approximation and may need adjustment based on your preprocessing
        num_end = len(numerical_features)
        cat_start = num_end
        
        feature_types[(0, num_end)] = 'numerical'
        feature_types[(cat_start, X_tensor.shape[1])] = 'categorical'
    elif numerical_features:
        feature_types[(0, X_tensor.shape[1])] = 'numerical'
    elif categorical_features:
        feature_types[(0, X_tensor.shape[1])] = 'categorical'
    
    # Handle NaN values based on feature types
    for range_or_idx, feat_type in feature_types.items():
        if isinstance(range_or_idx, tuple):
            # Range of columns
            start, end = range_or_idx
            if end > X_tensor.shape[1]:
                end = X_tensor.shape[1]
                
            X_subset = X_tensor[:, start:end]
            
            # Skip if no columns in this range
            if start >= X_subset.shape[1] or end <= 0:
                continue
                
            # Handle NaNs in this subset
            column_means = None
            if feat_type == 'numerical' and torch.isnan(X_subset).any():
                column_means = torch.nanmean(X_subset, dim=0, keepdim=True)
                
            X_subset = insure_none(X_subset, feature_type=feat_type, column_means=column_means)
            
            # Put back into X_tensor
            X_tensor[:, start:end] = X_subset
    
    # Make predictions
    with torch.no_grad():
        predictions = model(X_tensor)
    
    # Convert to numpy
    predictions_np = predictions.detach().numpy()
    
    # Process predictions based on approach
    if approach in ['dl_r', 'ml_r']:
        # Regression - inverse transform to get original scale
        predictions_original = target_transformer.inverse_transform(predictions_np)
        return predictions_original
    else:
        # Classification - convert to class labels
        if predictions_np.shape[1] > 1:
            # Multi-class: get class with highest probability
            pred_classes = np.argmax(predictions_np, axis=1)
            # Map indices to labels
            return [target_transformer[i] for i in pred_classes]
        else:
            # Binary classification
            pred_classes = (predictions_np > 0.5).astype(int)
            # Map 0/1 to actual class labels
            return [target_transformer[i[0]] for i in pred_classes]

# Example function to run the prediction pipeline
def predict_pipline(dataframe, model_path, preprocessor_dir, features,target, approach):
    """
    Load a model and make predictions on new data.
    
    Args:
        csv_path: Path to CSV with input data
        model_path: Path to saved model state_dict
        preprocessor_dir: Directory with preprocessors
        features: List of feature names
        approach: 'dl_r', 'ml_r', 'dl_c', or 'ml_c'
    
    Returns:
        DataFrame with original data and predictions
    """
    # Load the data
    df = dataframe
    
    # Check if all required features are in the dataframe
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in input data: {missing_features}")
    
    # Load model and preprocessors
    model, feature_preprocessor, target_transformer = load_model_for_prediction(
        model_path, preprocessor_dir, approach
    )
    
    # Make predictions
    predictions = predict_with_model(
        model, df, features, feature_preprocessor, target_transformer, approach
    )
    
    # Add predictions to dataframe
    if approach in ['dl_r', 'ml_r']:
        df[f'{target}'] = predictions
    else:
        df[f'{target}'] = predictions
    
    return df

def insure_none(x, feature_type='numerical', column_means=None, is_target=False):
    """
    Handle NaN values in tensor data by filling with appropriate values.
    
    Args:
        x: Tensor data potentially containing NaN values
        feature_type: Type of features ('numerical' or 'categorical')
        column_means: Precomputed column means for filling (if None, will compute from x)
        is_target: Whether this tensor is a target variable
        
    Returns:
        Tensor with NaN values replaced
    """
    if torch.isnan(x).any():
        if is_target:
            # For target variables, we might want special handling
            print("Warning: NaN values detected in target variable.")
            if feature_type == 'numerical':
                # For numerical targets, use mean imputation
                if column_means is None:
                    # Compute mean ignoring NaNs
                    column_means = torch.nanmean(x, dim=0, keepdim=True)
                
                print(f"Replacing NaN values in target with mean value: {column_means.item():.4f}")
                x = torch.nan_to_num(x, nan=column_means.item())
            else:
                # For categorical targets, this is more complex
                print("Warning: NaN values in categorical target may cause issues in model training.")
                # Fill with zeros, but this could be improved
                x = torch.nan_to_num(x, nan=0.0)
        else:
            # For feature variables
            if feature_type == 'numerical':
                # For numerical features, use column-wise mean imputation
                if column_means is None:
                    # Calculate mean for each feature, ignoring NaNs
                    column_means = torch.nanmean(x, dim=0, keepdim=True)
                
                # Create a mask of NaN values
                nan_mask = torch.isnan(x)
                
                # Fill NaN values with corresponding column means
                for col_idx in range(x.shape[1]):
                    col_mask = nan_mask[:, col_idx]
                    if col_mask.any():
                        x[col_mask, col_idx] = column_means[0, col_idx]
                
                print(f"Replaced NaN values in {torch.sum(nan_mask).item()} positions with column means.")
            
            elif feature_type == 'categorical':
                # For one-hot encoded categorical features
                # In one-hot encoding, NaN should translate to all zeros in that group
                nan_rows = torch.isnan(x).any(dim=1)
                if nan_rows.any():
                    print(f"Replaced NaN values in {torch.sum(nan_rows).item()} categorical instances with zeros.")
                    # Replace all NaNs with zeros (effectively making "unknown" category)
                    x = torch.nan_to_num(x, nan=0.0)
    
    return x

def return_regression_results(dataframe, features, target):
    """
    Run prediction pipeline and add statistical aggregation (mean/median) columns.
    
    Args:
        dataframe: Input DataFrame
        features: List of feature names
        target: Target column name
    
    Returns:
        DataFrame with predictions from ML/DL models and statistical aggregations
    """
    # Statistical metrics path
    path_statistical_metrics = "../server/results/stat_results/metrics.json"
    
    # Load mean and median values from JSON file
    try:
        with open(path_statistical_metrics, 'r') as f:
            stat_metrics = json.load(f)
            
        # Extract mean and median values
        if "Aggregated mean/median" in stat_metrics:
            mean_value = stat_metrics["Aggregated mean/median"][0]
            median_value = stat_metrics["Aggregated mean/median"][1]
            print(f"Loaded statistical metrics - Mean: {mean_value}, Median: {median_value}")
        else:
            print("Warning: 'Aggregated mean/median' not found in metrics JSON file")
            mean_value = None
            median_value = None
    except Exception as e:
        print(f"Error loading statistical metrics: {e}")
        mean_value = None
        median_value = None
    
    # Run prediction for ML and DL models
    for i in ['dl', 'ml']:
        model_path = f"../server/results/{i}_results/regression/agg_model/model_round100.pth"
        preprocessor_dir = f"../nodes/results/{i}_regression/"
        
        try:
            dataframe = predict_pipline(dataframe, model_path, preprocessor_dir, 
                                        features, target+f'_{i}', f"{i}_r")
        except Exception as e:
            print(f"Error during {i.upper()} prediction: {e}")
            # Create empty prediction column if prediction fails
            dataframe[target+f'_{i}'] = np.nan
    
    # Add mean and median columns if values were loaded successfully
    if mean_value is not None:
        dataframe[target+'_agg_mean'] = mean_value
        print(f"Added column '{target}_agg_mean' with value {mean_value}")
    
    if median_value is not None:
        dataframe[target+'_agg_median'] = median_value
        print(f"Added column '{target}_agg_median' with value {median_value}")
        
    # Calculate ensemble average of machine learning and deep learning predictions,
    # plus the aggregated mean and median values
    try:
        # Get values for each prediction method
        values_to_average = []
        columns_used = []
        
        # Add ML prediction if it exists
        if target+'_ml' in dataframe.columns:
            values_to_average.append(dataframe[target+'_ml'])
            columns_used.append(target+'_ml')
        
        # Add DL prediction if it exists
        if target+'_dl' in dataframe.columns:
            values_to_average.append(dataframe[target+'_dl'])
            columns_used.append(target+'_dl')
        
        # Add aggregated mean if it exists
        if target+'_agg_mean' in dataframe.columns:
            values_to_average.append(dataframe[target+'_agg_mean'])
            columns_used.append(target+'_agg_mean')
        
        # Add aggregated median if it exists
        if target+'_agg_median' in dataframe.columns:
            values_to_average.append(dataframe[target+'_agg_median'])
            columns_used.append(target+'_agg_median')
        
        # Calculate average if we have values
        if values_to_average:
            # Sum all values and divide by the number of columns
            dataframe[target+'_avg_predictions'] = sum(values_to_average) / len(values_to_average)
            
            print(f"Added column '{target}_avg_predictions' with average of: {', '.join(columns_used)}")
            
            # Verify calculation for the first row
            first_row_values = [val.iloc[0] for val in values_to_average]
            first_row_avg = sum(first_row_values) / len(first_row_values)
            print(f"First row values: {first_row_values}")
            print(f"First row average: {first_row_avg}")
        else:
            print(f"Warning: No prediction columns found for averaging")
            
    except Exception as e:
        print(f"Error calculating average predictions: {e}")
    
    return dataframe





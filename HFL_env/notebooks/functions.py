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
from scipy.stats import chi2_contingency, f_oneway
import os
from scipy import stats
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import math

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


def introduce_missingness(df, feature1, missing_rate, pattern='MCAR', feature2=None, task='regression', seed=42):
    """
    Introduce missing values in a specific feature of a DataFrame with a given pattern and rate,
    while saving the original values paired with appropriate time or ID index.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the data
        feature1 (str): Name of the feature to introduce missing values
        missing_rate (float): Rate of missing values to introduce (between 0 and 1)
        pattern (str): Pattern of missingness ('MCAR', 'MAR', or 'MNAR')
        feature2 (str, optional): Name of the target variable (needed for MAR pattern)
        task (str): Specifies the task type - 'regression' (uses 'charttime' as index) or 
                   'classification' (uses 'hadm_id' as index)
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple: (df_with_missing, ground_truth) where:
            - df_with_missing: DataFrame with missing values introduced
            - ground_truth: DataFrame with only original values and appropriate index
    """
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Check if feature exists in DataFrame
    if feature1 not in df.columns:
        raise ValueError(f"Feature '{feature1}' not found in DataFrame columns: {df.columns.tolist()}")
    
    # Determine index column based on task type
    index_col = 'charttime' if task == 'regression' else 'hadm_id'
    
    # Check if index column exists
    if index_col not in df.columns:
        raise ValueError(f"Column '{index_col}' not found in DataFrame. Required for task='{task}'")
    
    # Check if we need target for MAR pattern
    if pattern == 'MAR' and (feature2 is None or feature2 not in df.columns):
        raise ValueError(f"Target variable is required for MAR pattern. Target '{feature2}' not found.")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Create a ground truth DataFrame with only original values and the appropriate index
    ground_truth = pd.DataFrame(index=df_copy.index)
    ground_truth['original_values'] = df_copy[feature1].copy()
    ground_truth[index_col] = df_copy[index_col].copy()
    
    # Check for existing NaN values
    original_mask = df_copy[feature1].isna()
    existing_nan_count = original_mask.sum()
    n_samples = len(df_copy)
    existing_nan_rate = existing_nan_count / n_samples
    
    # Calculate how many values to make missing
    if existing_nan_rate > 0:
        print(f"Warning: Feature '{feature1}' already has {existing_nan_rate:.2%} missing values.")
        print("Using only rows with existing values to introduce new missingness.")
    
    # Focus only on non-missing values for introducing missingness
    valid_mask = ~df_copy[feature1].isna()
    valid_indices = df_copy.index[valid_mask].tolist()
    
    if len(valid_indices) == 0:
        raise ValueError(f"Feature '{feature1}' has no valid values. Cannot introduce missingness.")
    
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
            'target_value': df_copy.loc[valid_indices, feature2].values
        })
        
        # Sort by target values and select indices with lowest target values
        target_df = target_df.sort_values('target_value')
        missing_indices = target_df['index'].head(n_to_make_missing).tolist()
    
    elif pattern == 'MNAR':
        # Missing Not At Random - depends on feature value itself
        # Create a DataFrame with valid indices and feature values
        feature_df = pd.DataFrame({
            'index': valid_indices,
            'feature_value': df_copy.loc[valid_indices, feature1].values
        })
        
        # Sort by feature values and select indices with lowest feature values
        feature_df = feature_df.sort_values('feature_value')
        missing_indices = feature_df['index'].head(n_to_make_missing).tolist()
    
    else:
        raise ValueError("Pattern must be one of: 'MCAR', 'MAR', 'MNAR'")
    
    # Introduce missing values in the copy
    df_copy.loc[missing_indices, feature1] = np.nan
    
    # Calculate final missing rate for verification
    final_missing_count = df_copy[feature1].isna().sum()
    final_missing_rate = final_missing_count / n_samples
    
    print(f"Task: {task} (using {index_col} as index)")
    print(f"Missingness pattern: {pattern}")
    print(f"Original missing rate: {existing_nan_rate:.2%}")
    print(f"Added {n_to_make_missing} missing values")
    print(f"Final missing rate: {final_missing_rate:.2%}")
    
    return df_copy, ground_truth


def prepare_clean_dataset(df, target=None, features=None):
    """
    Create a new DataFrame with only rows that have non-missing values for both specified features.
    
    Args:
        df (pandas.DataFrame): Input DataFrame
        feature1 (str): First feature name to check for missing values
        feature2 (str): Second feature name to check for missing values
        
    Returns:
        pandas.DataFrame: Clean dataset with no missing values in both specified features
    """
    if features is not None :
        clean_df = df.dropna(subset=features)
    else :
        clean_df = df.dropna(subset=[target])
  
    
    
    # Report statistics
    original_count = len(df)
    clean_count = len(clean_df)
    
    print(f"Original dataset: {original_count} rows")
    print(f"Clean dataset: {clean_count} rows ({clean_count/original_count:.2%} of original)")
    print(f"\nMissing values summary:")
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

def return_classification_results(dataframe, features):
    """
    Run prediction pipeline and add statistical mode column.
    
    Args:
        dataframe: Input DataFrame
        features: List of feature names
        target: Target column name
    
    Returns:
        DataFrame with predictions from ML/DL models and statistical aggregations
    """
    target = features[0]
    features=features[1:]
    # Statistical mode
    statistical_mode = dataframe[target].mode()[0]
    
    # Run prediction for ML and DL models
    for i in ['dl', 'ml']:
        model_path = f"../server/results/{i}_results/classification/agg_model/model_round100.pth"
        preprocessor_dir = f"../nodes/results/{i}_classification/"
        
        try:
            dataframe = predict_pipline(dataframe, model_path, preprocessor_dir, 
                                        features, target+f'_{i}', f"{i}_c")
        except Exception as e:
            print(f"Error during {i.upper()} prediction: {e}")
            # Create empty prediction column if prediction fails
            dataframe[target+f'_{i}'] = np.nan
    
    # Add mode column if values were loaded successfully
    if statistical_mode is not None:
        dataframe[target+'_mode'] = statistical_mode

    return dataframe

def benchmark_predictions(pred_df, original_df, target, time_col='charttime'):
    """
    Perform detailed benchmarking of predicted values against original values.
    
    Args:
        pred_df: DataFrame containing predictions from various approaches
        original_df: DataFrame containing original values and charttime
        target: Name of the target variable (without suffixes)
        time_col: Name of the time column for merging
    
    Returns:
        Dictionary with benchmark results
    """

    path = "./benchmark_results"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    print(f"{'='*80}")
    print(f"BENCHMARKING PREDICTIONS FOR {target.upper()}")
    print(f"{'='*80}")
    
    # Merge prediction dataframe with original values
    benchmark_df = pd.merge(pred_df, original_df, on=time_col, how='inner')
    
    # Identify all prediction columns (those starting with target_)
    pred_columns = [col for col in benchmark_df.columns if col.startswith(target+'_')]
    
    print(f"\nFound {len(pred_columns)} prediction approaches:")
    for col in pred_columns:
        print(f"  - {col}")
    
    print(f"\nAnalyzing {len(benchmark_df)} rows with matching timestamps.")
    
    # Global metrics table
    print(f"\n{'-'*80}")
    print("GLOBAL PERFORMANCE METRICS")
    print(f"{'-'*80}")
    
    metrics = {}
    metrics_df = pd.DataFrame(
        columns=['Approach', 'RMSE', 'MAE', 'MAPE (%)', 'R²', 'Explained Variance', 
                 'Mean Error', 'Error Std', 'Correlation']
    )
    
    # Calculate metrics for each prediction approach
    for i, col in enumerate(pred_columns):
        # Calculate error metrics
        true = benchmark_df['original_values']
        pred = benchmark_df[col]
        
        rmse = math.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        
        # Calculate MAPE with handling for zero values
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((true - pred) / true)) * 100
            if np.isnan(mape) or np.isinf(mape):
                # Filter out zeros before calculating MAPE
                nonzero_idx = true != 0
                if nonzero_idx.sum() > 0:
                    mape = np.mean(np.abs((true[nonzero_idx] - pred[nonzero_idx]) / true[nonzero_idx])) * 100
                else:
                    mape = np.nan
        
        r2 = r2_score(true, pred)
        exp_var = explained_variance_score(true, pred)
        
        # Error analysis
        errors = true - pred
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Correlation
        correlation, _ = stats.pearsonr(true, pred)
        
        # Store metrics
        approach_name = col.replace(target+'_', '')
        metrics[approach_name] = {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'exp_var': exp_var,
            'mean_error': mean_error,
            'std_error': std_error,
            'correlation': correlation,
            'predictions': pred,
            'true_values': true
        }
        
        # Add to metrics dataframe
        metrics_df.loc[i] = [
            approach_name, rmse, mae, f"{mape:.2f}", r2, exp_var, 
            mean_error, std_error, correlation
        ]
    
    # Sort metrics by RMSE (best performers first)
    metrics_df = metrics_df.sort_values('RMSE')
    
    # Display metrics table
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(metrics_df)
    
    # Identify best approach
    best_approach = metrics_df.iloc[0]['Approach']
    print(f"\nBest performing approach: {best_approach} (RMSE: {metrics_df.iloc[0]['RMSE']:.4f})")
    
    # Create visualizations
    print(f"\n{'-'*80}")
    print("VISUALIZING PREDICTIONS VS ACTUAL VALUES")
    print(f"{'-'*80}")
    
    # 1. Actual vs Predicted scatter plots
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'grey']
    plt.figure(figsize=(18, 10))
    
    for i, approach_name in enumerate(metrics.keys()):
        plt.subplot(2, (len(metrics) + 1) // 2, i+1)
        
        true = metrics[approach_name]['true_values']
        pred = metrics[approach_name]['predictions']
        
        plt.scatter(true, pred, alpha=0.6, color=colors[i % len(colors)])
        
        # Add perfect prediction line
        min_val = min(true.min(), pred.min())
        max_val = max(true.max(), pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add regression line
        z = np.polyfit(true, pred, 1)
        p = np.poly1d(z)
        plt.plot(true, p(true), color=colors[i % len(colors)], linestyle='-')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{approach_name}: R² = {metrics[approach_name]["r2"]:.4f}')
        
        # Add RMSE and correlation to plot
        plt.annotate(f'RMSE: {metrics[approach_name]["rmse"]:.4f}\nCorr: {metrics[approach_name]["correlation"]:.4f}', 
                   xy=(0.05, 0.85), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig(f'./benchmark_results/{target}_predictions_scatter.png', dpi=300)
    plt.show()
    
    # 2. Error distribution plots
    plt.figure(figsize=(18, 10))
    
    for i, approach_name in enumerate(metrics.keys()):
        plt.subplot(2, (len(metrics) + 1) // 2, i+1)
        
        true = metrics[approach_name]['true_values']
        pred = metrics[approach_name]['predictions']
        errors = true - pred
        
        sns.histplot(errors, kde=True, color=colors[i % len(colors)])
        plt.axvline(x=0, color='r', linestyle='--')
        plt.xlabel('Error (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.title(f'{approach_name}: Error Distribution')
        
        # Add error stats to plot
        plt.annotate(f'Mean: {metrics[approach_name]["mean_error"]:.4f}\nStd: {metrics[approach_name]["std_error"]:.4f}', 
                   xy=(0.05, 0.85), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig(f'./benchmark_results/{target}_error_distribution.png', dpi=300)
    plt.show()
    
    # 3. Time series plot of actual vs best predictions
    plt.figure(figsize=(15, 8))
    
    # Sort by timestamp
    time_series_df = benchmark_df.sort_values(by=time_col)
    
    # Get the best approach based on RMSE
    best_pred_col = target + '_' + best_approach
    
    plt.plot(time_series_df[time_col], time_series_df['original_values'], 'b-', label='Actual Values')
    plt.plot(time_series_df[time_col], time_series_df[best_pred_col], 'r--', label=f'Predicted ({best_approach})')
    
    plt.xlabel('Time')
    plt.ylabel(target)
    plt.title(f'Actual vs Best Predictions ({best_approach}) Over Time')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./benchmark_results/{target}_time_series.png', dpi=300)
    plt.show()
    
    # 4. Boxplot of errors by approach
    plt.figure(figsize=(12, 8))
    
    error_data = []
    for approach_name in metrics.keys():
        pred_col = target + '_' + approach_name
        errors = benchmark_df['original_values'] - benchmark_df[pred_col]
        for error in errors:
            error_data.append({
                'Approach': approach_name,
                'Error': error
            })
    
    error_df = pd.DataFrame(error_data)
    
    sns.boxplot(x='Approach', y='Error', data=error_df)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Error Distribution by Approach')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'./benchmark_results/{target}_error_boxplot.png', dpi=300)
    plt.show()
    
    # 5. Heatmap of prediction correlations
    plt.figure(figsize=(10, 8))
    
    pred_cols = [target + '_' + approach for approach in metrics.keys()]
    corr_matrix = benchmark_df[pred_cols + ['original_values']].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Between Different Prediction Approaches')
    plt.tight_layout()
    plt.savefig(f'./benchmark_results/{target}_prediction_correlations.png', dpi=300)
    plt.show()
    
    # Detailed error analysis table
    print(f"\n{'-'*80}")
    print("DETAILED ERROR ANALYSIS")
    print(f"{'-'*80}")
    
    # Create bins for error analysis
    error_analysis = {}
    
    for approach_name in metrics.keys():
        pred_col = target + '_' + approach_name
        
        # Calculate percent errors
        true = benchmark_df['original_values']
        pred = benchmark_df[pred_col]
        abs_errors = np.abs(true - pred)
        
        # Count errors within different ranges
        error_ranges = {
            '< 5%': np.mean(abs_errors / true < 0.05) * 100,
            '5-10%': np.mean((abs_errors / true >= 0.05) & (abs_errors / true < 0.10)) * 100,
            '10-20%': np.mean((abs_errors / true >= 0.10) & (abs_errors / true < 0.20)) * 100,
            '> 20%': np.mean(abs_errors / true >= 0.20) * 100
        }
        
        error_analysis[approach_name] = error_ranges
    
    error_analysis_df = pd.DataFrame(error_analysis).T
    print("Percentage of predictions within error ranges:")
    print(error_analysis_df)
    
    # Summary of results
    print(f"\n{'-'*80}")
    print("SUMMARY OF FINDINGS")
    print(f"{'-'*80}")
    
    # Sort approaches by RMSE
    approaches_by_rmse = metrics_df.sort_values('RMSE')['Approach'].tolist()
    
    print(f"1. Best performing approach: {best_approach}")
    print(f"2. Ranking of approaches by RMSE (best to worst):")
    for i, approach in enumerate(approaches_by_rmse):
        print(f"   {i+1}. {approach} (RMSE: {metrics_df[metrics_df['Approach'] == approach]['RMSE'].values[0]:.4f})")
    
    print(f"3. The best approach ({best_approach}) has:")
    print(f"   - {error_analysis_df.loc[best_approach]['< 5%']:.1f}% of predictions within 5% of actual values")
    print(f"   - {error_analysis_df.loc[best_approach]['< 5%'] + error_analysis_df.loc[best_approach]['5-10%']:.1f}% of predictions within 10% of actual values")


    # Create a copy of detailed_metrics without predictions and true_values
    cleaned_detailed_metrics = {}
    for approach, details in metrics.items():
        cleaned_details = {k: v for k, v in details.items() if k not in ['predictions', 'true_values']}
        cleaned_detailed_metrics[approach] = cleaned_details

    # Create a cleaned results dict
    results_dict = {
        'metrics': metrics_df,
        'detailed_metrics': cleaned_detailed_metrics,
        'error_analysis': error_analysis_df,
        'best_approach': best_approach
    }

    # Make it all serializable
    serializable_results = make_serializable(results_dict)

    # Save as JSON
    with open(path + '/summary.json', 'w') as f:
        json.dump(serializable_results, f, indent=2)

def get_nan_rows(dataframe, features):
    """
    Return all rows where the specified feature(s) has NaN values.
    
    Args:
        dataframe: pandas DataFrame
        features: str or list of str - column name(s) to check for NaN values
    
    Returns:
        pandas DataFrame containing only rows where at least one of the specified features has NaN
    """
    # Convert single feature to list for consistent handling
    if isinstance(features, str):
        features = [features]
    
    # Validate that all features exist in the dataframe
    missing_features = [f for f in features if f not in dataframe.columns]
    if missing_features:
        raise ValueError(f"Features not found in dataframe: {', '.join(missing_features)}")
    
    # Create a mask for rows where any of the specified features is NaN
    nan_mask = dataframe[features].isna().any(axis=1)
    
    # Get the count of NaN rows
    nan_count = nan_mask.sum()
    
    # Filter the dataframe using the mask
    nan_rows = dataframe[nan_mask].copy()
    
    # Print summary
    print(f"Found {nan_count} rows with NaN values in features: {', '.join(features)}")
    print(f"This represents {(nan_count / len(dataframe) * 100):.2f}% of the dataframe ({nan_count} out of {len(dataframe)} rows)")
    
    # Count NaNs per feature
    if len(features) > 1:
        print("\nNaN count by feature:")
        for feature in features:
            feature_nan_count = dataframe[feature].isna().sum()
            print(f"  - {feature}: {feature_nan_count} NaN values ({(feature_nan_count / len(dataframe) * 100):.2f}%)")
    
    return nan_rows

def make_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(i) for i in obj]
    else:
        return obj


def _interpret_eta_squared(eta_squared):
    """Interpret eta squared value"""
    if eta_squared < 0.01:
        return "Negligible"
    elif eta_squared < 0.06:
        return "Small"
    elif eta_squared < 0.14:
        return "Medium"
    else:
        return "Large"

def _interpret_cramers_v(v):
    """Interpret Cramer's V value"""
    if v < 0.1:
        return "Negligible"
    elif v < 0.2:
        return "Weak"
    elif v < 0.3:
        return "Moderate"
    elif v < 0.4:
        return "Relatively Strong"
    else:
        return "Strong"

def identify_significant_features(df, target_col, significance_threshold=0.05, 
                                  cramers_v_threshold=0.1, eta_squared_threshold=0.01,
                                  top_n=None, show_plots=False):
    """
    Identifies the most significant features for predicting a categorical target
    based on statistical metrics (p-values, Cramer's V, and eta squared).
    
    Args:
        df: DataFrame containing the data
        target_col: Categorical target column
        significance_threshold: p-value threshold for statistical significance (default: 0.05)
        cramers_v_threshold: minimum Cramer's V value for categorical features (default: 0.1)
        eta_squared_threshold: minimum eta squared value for numerical features (default: 0.01)
        top_n: optional limit on number of features to return (default: None = all significant)
        show_plots: whether to display distribution plots (default: False)
    
    Returns:
        DataFrame containing the most significant features and their metrics
    """
    # Check if target column exists
    if target_col not in df.columns:
        print(f"Error: {target_col} not found in dataframe")
        return None
    
    # Get all feature columns (excluding target)
    feature_cols = [col for col in df.columns if col != target_col]
    
    if not feature_cols:
        print("No features found to analyze")
        return None
    
    print(f"Analyzing {len(feature_cols)} features to identify predictors for '{target_col}'")
    
    # Store all feature results
    all_features_data = []
    
    # Process each feature
    for col in feature_cols:
        # Create a clean subset with this pair
        pair_df = df[[col, target_col]].dropna()
        valid_count = len(pair_df)
        total_count = len(df)
        
        if valid_count < 5:
            continue
        
        # Determine if feature is categorical or numerical
        is_categorical = False
        
        # Check if column is categorical-like
        if not pd.api.types.is_numeric_dtype(df[col]):
            is_categorical = True
        # Or if it's a numeric column with few unique values (likely categorical)
        elif df[col].nunique() <= 10:
            is_categorical = True
        
        # Analysis based on feature type
        feature_data = {
            'Feature': col,
            'Type': 'categorical' if is_categorical else 'numerical',
            'Valid Observations': valid_count,
            'Data Coverage': f"{valid_count/total_count:.1%}"
        }
        
        if is_categorical:
            metrics = _analyze_cat_feature_metrics(pair_df, col, target_col)
            if metrics:
                feature_data.update({
                    'Test': 'Chi-square',
                    'Statistic': metrics['chi2'],
                    'p-value': metrics['p_value'],
                    'Effect Size': metrics['cramers_v'],
                    'Effect Size Name': "Cramer's V",
                    'Interpretation': metrics['interpretation'],
                    'Significant': metrics['p_value'] <= significance_threshold,
                    'Strong Enough': metrics['cramers_v'] >= cramers_v_threshold,
                    'Low Expected Freq': metrics['low_expected_freq'],
                    'DoF': metrics['dof']
                })
                all_features_data.append(feature_data)
        else:
            metrics = _analyze_num_feature_metrics(pair_df, col, target_col, False)
            if metrics and 'error' not in metrics:
                feature_data.update({
                    'Test': 'ANOVA',
                    'Statistic': metrics['f_stat'],
                    'p-value': metrics['p_value'],
                    'Effect Size': metrics['eta_squared'],
                    'Effect Size Name': "Eta squared",
                    'Interpretation': metrics['interpretation'],
                    'Significant': metrics['p_value'] <= significance_threshold,
                    'Strong Enough': metrics['eta_squared'] >= eta_squared_threshold,
                    'Low Expected Freq': False,
                    'DoF': None
                })
                all_features_data.append(feature_data)
    
    # If no valid features found
    if not all_features_data:
        print("No valid features found for analysis")
        return None
        
    # Create DataFrame with all features
    all_features_df = pd.DataFrame(all_features_data)
    
    # Sort by significance and effect size
    all_features_df['Recommended'] = all_features_df['Significant'] & all_features_df['Strong Enough']
    all_features_df = all_features_df.sort_values(['Recommended', 'Effect Size'], ascending=[False, False])
    
    # Format the full results table
    display_cols = ['Feature', 'Type', 'Test', 'Statistic', 'p-value', 
                    'Effect Size', 'Effect Size Name', 'Interpretation',
                    'Significant', 'Strong Enough', 'Recommended', 
                    'Valid Observations', 'Data Coverage']
    
    if 'Low Expected Freq' in all_features_df.columns:
        display_cols.insert(display_cols.index('Valid Observations'), 'Low Expected Freq')
    
    full_results = all_features_df[display_cols].copy()
    
    # Filter for significant features
    sig_features = all_features_df[all_features_df['Recommended']].copy()
    
    # Apply top_n limit if specified
    if top_n is not None and top_n > 0 and len(sig_features) > top_n:
        sig_features = sig_features.head(top_n)
    
    # Display the results table
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    
    print("\n=========================================== Full Analysis Results =========================================== ")
    print(full_results)
    
    print("\n===========================================  RECOMMENDED FEATURES FOR PREDICTING", target_col.upper(), "=========================================== ")
    if len(sig_features) > 0:
        # Simplify the recommended features display
        recommended_cols = ['Feature', 'Type', 'Effect Size', 'Effect Size Name', 
                           'p-value', 'Interpretation']
        print(sig_features[recommended_cols])
    else:
        print("No features meet the significance and effect size criteria.")
        print(f"Consider relaxing the thresholds (p-value < {significance_threshold}, effect size thresholds).")
    
def _analyze_num_feature_metrics(df, feature_col, target_col, show_plot=False):
    """Get metrics for numerical feature vs categorical target"""
    try:
        # ANOVA test
        groups = [df[df[target_col] == cat][feature_col].values for cat in df[target_col].unique()]
        
        if len(groups) < 2 or not all(len(g) > 1 for g in groups):
            return {
                'error': 'insufficient_groups',
                'mean': df[feature_col].mean(),
                'std': df[feature_col].std()
            }
        
        # Run ANOVA
        f_stat, p_value = f_oneway(*groups)
        
        # Calculate eta squared (effect size)
        grand_mean = df[feature_col].mean()
        total_ss = sum((df[feature_col] - grand_mean) ** 2)
        between_ss = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        eta_squared = between_ss / total_ss if total_ss > 0 else 0
        
        result = {
            'f_stat': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'interpretation': _interpret_eta_squared(eta_squared)
        }
        
        # Distribution plot if requested (disabled by default)
        if show_plot and p_value <= 0.05 and eta_squared >= 0.01:
            plt.figure(figsize=(10, 6))
            
            # Plot KDE for each category
            for category in df[target_col].unique():
                subset = df[df[target_col] == category]
                sns.kdeplot(subset[feature_col], label=f"{target_col}={category}")
            
            plt.title(f'Distribution of {feature_col} by {target_col}')
            plt.xlabel(feature_col)
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.show()
        
        return result
    except:
        return None

def _analyze_cat_feature_metrics(df, feature_col, target_col):
    """Get metrics for categorical feature vs categorical target"""
    try:
        # Calculate contingency table
        cross_tab = pd.crosstab(df[feature_col], df[target_col])
        
        # Chi-square test
        chi2, p, dof, expected = chi2_contingency(cross_tab)
        
        # Calculate Cramer's V
        n = cross_tab.sum().sum()
        phi2 = chi2 / n
        r, k = cross_tab.shape
        phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        r_corr = r - (r-1)**2/(n-1)
        k_corr = k - (k-1)**2/(n-1)
        cramers_v = np.sqrt(phi2_corr / min(k_corr-1, r_corr-1)) if min(k_corr-1, r_corr-1) > 0 else 0
        
        # Check for low expected frequencies
        low_exp_freq = (expected < 5).any()
        
        return {
            'chi2': chi2,
            'p_value': p,
            'dof': dof,
            'cramers_v': cramers_v,
            'interpretation': _interpret_cramers_v(cramers_v),
            'low_expected_freq': low_exp_freq
        }
    except:
        return None

   








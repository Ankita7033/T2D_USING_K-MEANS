"""
FCDT-TPFF: Temporal Validation Framework for JBI
✓ Temporal cross-validation
✓ Temporal distribution shift detection
✓ Prospective validation simulation
✓ Dataset drift quantification
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.spatial.distance import jensenshannon
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class TemporalValidator:
    """
    Temporal validation for time-series clinical data
    """
    
    def __init__(self, time_column='admittime', random_state=42):
        self.time_column = time_column
        self.random_state = random_state
        np.random.seed(random_state)
    
    # =========================================================================
    # TEMPORAL CROSS-VALIDATION
    # =========================================================================
    
    def temporal_split(self, data, test_fraction=0.2, n_splits=5, gap_days=0):
        """
        Create temporal train/test splits
        
        Args:
            data: DataFrame with time column
            test_fraction: Fraction of data for testing
            n_splits: Number of temporal splits
            gap_days: Gap between train and test (to avoid leakage)
            
        Returns:
            list: List of (train_idx, test_idx) tuples
        """
        print("\n" + "="*60)
        print("TEMPORAL CROSS-VALIDATION SPLITS")
        print("="*60)
        
        # Sort by time
        data_sorted = data.sort_values(self.time_column).reset_index(drop=True)
        
        if self.time_column not in data_sorted.columns:
            raise ValueError(f"Time column '{self.time_column}' not found")
        
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(data_sorted[self.time_column]):
            data_sorted[self.time_column] = pd.to_datetime(data_sorted[self.time_column])
        
        n_total = len(data_sorted)
        n_test = int(n_total * test_fraction)
        
        splits = []
        
        for i in range(n_splits):
            # Progressive temporal splits
            # Each split uses earlier data for training, later for testing
            test_end = n_total - i * (n_test // n_splits)
            test_start = test_end - n_test
            
            if test_start < n_total * 0.3:  # Need minimum training data
                break
            
            # Apply gap
            if gap_days > 0:
                gap_mask = (
                    data_sorted[self.time_column] >= 
                    data_sorted[self.time_column].iloc[test_start] - pd.Timedelta(days=gap_days)
                ) & (
                    data_sorted[self.time_column] < 
                    data_sorted[self.time_column].iloc[test_start]
                )
                
                train_idx = data_sorted.index[
                    ~((data_sorted.index >= test_start) | gap_mask)
                ]
            else:
                train_idx = data_sorted.index[:test_start]
            
            test_idx = data_sorted.index[test_start:test_end]
            
            # Print split info
            train_dates = data_sorted.loc[train_idx, self.time_column]
            test_dates = data_sorted.loc[test_idx, self.time_column]
            
            print(f"\nSplit {i+1}:")
            print(f"  Train: {len(train_idx)} samples")
            print(f"    Date range: {train_dates.min().date()} to {train_dates.max().date()}")
            print(f"  Test: {len(test_idx)} samples")
            print(f"    Date range: {test_dates.min().date()} to {test_dates.max().date()}")
            print(f"  Gap: {gap_days} days")
            
            splits.append((train_idx, test_idx))
        
        return splits
    
    def single_temporal_split(self, data, train_end_date=None, test_fraction=0.3):
        """
        Single temporal split (most common for publication)
        
        Args:
            data: DataFrame with time column
            train_end_date: Cutoff date (if None, uses quantile)
            test_fraction: Fraction for test set if no date provided
            
        Returns:
            tuple: (train_idx, test_idx)
        """
        data_sorted = data.sort_values(self.time_column).reset_index(drop=True)
        
        if not pd.api.types.is_datetime64_any_dtype(data_sorted[self.time_column]):
            data_sorted[self.time_column] = pd.to_datetime(data_sorted[self.time_column])
        
        if train_end_date is None:
            # Use quantile
            cutoff = data_sorted[self.time_column].quantile(1 - test_fraction)
        else:
            cutoff = pd.to_datetime(train_end_date)
        
        train_mask = data_sorted[self.time_column] <= cutoff
        test_mask = data_sorted[self.time_column] > cutoff
        
        train_idx = data_sorted[train_mask].index
        test_idx = data_sorted[test_mask].index
        
        print("\n" + "="*60)
        print("TEMPORAL TRAIN/TEST SPLIT")
        print("="*60)
        print(f"Cutoff date: {cutoff.date()}")
        print(f"Training: {len(train_idx)} samples ({len(train_idx)/len(data)*100:.1f}%)")
        print(f"  Date range: {data_sorted.loc[train_idx, self.time_column].min().date()} to {data_sorted.loc[train_idx, self.time_column].max().date()}")
        print(f"Testing: {len(test_idx)} samples ({len(test_idx)/len(data)*100:.1f}%)")
        print(f"  Date range: {data_sorted.loc[test_idx, self.time_column].min().date()} to {data_sorted.loc[test_idx, self.time_column].max().date()}")
        
        return train_idx, test_idx
    
    # =========================================================================
    # TEMPORAL DRIFT DETECTION
    # =========================================================================
    
    def detect_distribution_shift(self, X_train, X_test, feature_names=None):
        """
        Quantify temporal distribution shift using multiple metrics
        
        Args:
            X_train: Training data (numpy array or DataFrame)
            X_test: Test data
            feature_names: List of feature names
            
        Returns:
            DataFrame: Shift metrics for each feature
        """
        print("\n" + "="*60)
        print("TEMPORAL DISTRIBUTION SHIFT ANALYSIS")
        print("="*60)
        
        if isinstance(X_train, pd.DataFrame):
            feature_names = X_train.columns.tolist()
            X_train = X_train.values
            X_test = X_test.values
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        results = []
        
        for i, feat_name in enumerate(feature_names):
            train_feat = X_train[:, i]
            test_feat = X_test[:, i]
            
            # Remove NaN
            train_feat = train_feat[~np.isnan(train_feat)]
            test_feat = test_feat[~np.isnan(test_feat)]
            
            if len(train_feat) < 10 or len(test_feat) < 10:
                continue
            
            # 1. Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(train_feat, test_feat)
            
            # 2. Mean/std shift
            mean_shift = (np.mean(test_feat) - np.mean(train_feat)) / (np.std(train_feat) + 1e-8)
            std_ratio = np.std(test_feat) / (np.std(train_feat) + 1e-8)
            
            # 3. Population Stability Index (PSI)
            psi = self.calculate_psi(train_feat, test_feat)
            
            # Interpretation
            if psi < 0.1:
                stability = "stable"
            elif psi < 0.25:
                stability = "moderate shift"
            else:
                stability = "significant shift"
            
            results.append({
                'feature': feat_name,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'mean_shift_sd': mean_shift,
                'std_ratio': std_ratio,
                'psi': psi,
                'stability': stability
            })
        
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('psi', ascending=False)
        
        # Print summary
        print(f"\nAnalyzed {len(results_df)} features")
        print(f"Significant shifts (KS p<0.05): {(results_df['ks_pvalue'] < 0.05).sum()}")
        print(f"High PSI (>0.25): {(results_df['psi'] > 0.25).sum()}")
        
        print("\nTop 5 features with largest shifts:")
        print(results_df[['feature', 'psi', 'mean_shift_sd', 'stability']].head().to_string(index=False))
        
        return results_df
    
    def calculate_psi(self, expected, actual, bins=10):
        """
        Calculate Population Stability Index
        
        PSI < 0.1: No significant shift
        PSI 0.1-0.25: Moderate shift
        PSI > 0.25: Significant shift
        """
        # Create bins
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins+1))
        breakpoints = np.unique(breakpoints)  # Remove duplicates
        
        if len(breakpoints) < 3:
            return 0.0  # Not enough variation
        
        # Bin the data
        expected_bins = np.digitize(expected, breakpoints[1:-1])
        actual_bins = np.digitize(actual, breakpoints[1:-1])
        
        # Count in each bin
        expected_counts = np.bincount(expected_bins, minlength=len(breakpoints))
        actual_counts = np.bincount(actual_bins, minlength=len(breakpoints))
        
        # Convert to percentages
        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)
        
        # Avoid division by zero
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return psi
    
    # =========================================================================
    # MODEL PERFORMANCE ACROSS TIME
    # =========================================================================
    
    def evaluate_temporal_performance(self, model, splits, X, y=None, 
                                     metric='silhouette', output_path="./processed_data"):
        """
        Evaluate model performance across temporal splits
        
        Args:
            model: Trained clustering model
            splits: List of (train_idx, test_idx) tuples
            X: Feature matrix
            y: True labels (if available)
            metric: Evaluation metric
            
        Returns:
            DataFrame: Performance metrics for each split
        """
        print("\n" + "="*60)
        print("TEMPORAL PERFORMANCE EVALUATION")
        print("="*60)
        
        results = []
        
        for i, (train_idx, test_idx) in enumerate(splits):
            X_train = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
            X_test = X[test_idx] if isinstance(X, np.ndarray) else X.iloc[test_idx]
            
            # Get predictions
            train_labels = model.predict(X_train)
            test_labels = model.predict(X_test)
            
            # Calculate metrics
            if metric == 'silhouette':
                train_score = silhouette_score(X_train, train_labels)
                test_score = silhouette_score(X_test, test_labels)
            
            # Performance decay
            decay = train_score - test_score
            
            results.append({
                'split': i+1,
                'train_score': train_score,
                'test_score': test_score,
                'performance_decay': decay,
                'n_train': len(train_idx),
                'n_test': len(test_idx)
            })
            
            print(f"\nSplit {i+1}:")
            print(f"  Train {metric}: {train_score:.4f}")
            print(f"  Test {metric}: {test_score:.4f}")
            print(f"  Decay: {decay:.4f}")
        
        results_df = pd.DataFrame(results)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['split'], results_df['train_score'], 'o-', label='Training', linewidth=2)
        plt.plot(results_df['split'], results_df['test_score'], 's-', label='Testing', linewidth=2)
        plt.xlabel('Temporal Split')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title('Model Performance Across Time')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_path}/temporal_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Summary
        print("\n" + "-"*60)
        print("SUMMARY")
        print("-"*60)
        print(f"Mean train {metric}: {results_df['train_score'].mean():.4f} ± {results_df['train_score'].std():.4f}")
        print(f"Mean test {metric}: {results_df['test_score'].mean():.4f} ± {results_df['test_score'].std():.4f}")
        print(f"Mean performance decay: {results_df['performance_decay'].mean():.4f}")
        
        if results_df['performance_decay'].mean() > 0.1:
            print("\n⚠ WARNING: Significant performance decay detected")
            print("  Consider domain adaptation or periodic retraining")
        else:
            print("\n✓ Performance stable across time")
        
        results_df.to_csv(f"{output_path}/temporal_performance.csv", index=False)
        
        return results_df
    
    # =========================================================================
    # PROSPECTIVE VALIDATION SIMULATION
    # =========================================================================
    
    def simulate_prospective_validation(self, data, X, model, cutoff_date, 
                                       lookback_days=365, output_path="./processed_data"):
        """
        Simulate prospective validation
        
        Train on historical data, validate on future data
        
        Args:
            data: DataFrame with time column and features
            X: Feature matrix (must align with data)
            model: Clustering model
            cutoff_date: Date to split train/prospective
            lookback_days: How far back to use for training
            
        Returns:
            dict: Prospective validation results
        """
        print("\n" + "="*60)
        print("PROSPECTIVE VALIDATION SIMULATION")
        print("="*60)
        
        data = data.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(data[self.time_column]):
            data[self.time_column] = pd.to_datetime(data[self.time_column])
        
        cutoff_date = pd.to_datetime(cutoff_date)
        lookback_date = cutoff_date - pd.Timedelta(days=lookback_days)
        
        # Define cohorts
        train_mask = (data[self.time_column] >= lookback_date) & (data[self.time_column] < cutoff_date)
        prospective_mask = data[self.time_column] >= cutoff_date
        
        train_idx = data[train_mask].index
        prosp_idx = data[prospective_mask].index
        
        print(f"\nCutoff date: {cutoff_date.date()}")
        print(f"Lookback: {lookback_days} days")
        print(f"\nTraining cohort:")
        print(f"  Date range: {data.loc[train_idx, self.time_column].min().date()} to {data.loc[train_idx, self.time_column].max().date()}")
        print(f"  N = {len(train_idx)}")
        print(f"\nProspective cohort:")
        print(f"  Date range: {data.loc[prosp_idx, self.time_column].min().date()} to {data.loc[prosp_idx, self.time_column].max().date()}")
        print(f"  N = {len(prosp_idx)}")
        
        # Get data
        X_train = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
        X_prosp = X[prosp_idx] if isinstance(X, np.ndarray) else X.iloc[prosp_idx]
        
        # Train model
        print("\nTraining on historical data...")
        model.fit(X_train)
        
        # Predict on prospective
        print("Validating on prospective data...")
        train_labels = model.predict(X_train)
        prosp_labels = model.predict(X_prosp)
        
        # Evaluate
        train_sil = silhouette_score(X_train, train_labels)
        prosp_sil = silhouette_score(X_prosp, prosp_labels)
        
        print("\n" + "-"*60)
        print("RESULTS")
        print("-"*60)
        print(f"Training silhouette: {train_sil:.4f}")
        print(f"Prospective silhouette: {prosp_sil:.4f}")
        print(f"Performance decay: {train_sil - prosp_sil:.4f}")
        
        # Distribution shift
        shift_results = self.detect_distribution_shift(X_train, X_prosp)
        
        # Save
        results = {
            'train_silhouette': train_sil,
            'prospective_silhouette': prosp_sil,
            'performance_decay': train_sil - prosp_sil,
            'n_train': len(train_idx),
            'n_prospective': len(prosp_idx),
            'distribution_shift': shift_results
        }
        
        shift_results.to_csv(f"{output_path}/prospective_shift.csv", index=False)
        
        return results
    
    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================
    
    def generate_temporal_validation_report(self, data, X, model=None, 
                                           test_fraction=0.3, output_path="./processed_data"):
        """
        Generate comprehensive temporal validation report
        
        This is the main function to call for JBI-ready temporal validation
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE TEMPORAL VALIDATION REPORT")
        print("="*80)
        
        report = {}
        
        # 1. Single temporal split
        train_idx, test_idx = self.single_temporal_split(data, test_fraction=test_fraction)
        report['split'] = {'train_idx': train_idx, 'test_idx': test_idx}
        
        # 2. Distribution shift analysis
        X_train = X.iloc[train_idx] if isinstance(X, pd.DataFrame) else X[train_idx]
        X_test = X.iloc[test_idx] if isinstance(X, pd.DataFrame) else X[test_idx]
        
        report['distribution_shift'] = self.detect_distribution_shift(X_train, X_test)
        report['distribution_shift'].to_csv(f"{output_path}/temporal_shift.csv", index=False)
        
        # 3. Visualize shifts
        self.plot_temporal_shifts(data, train_idx, test_idx, output_path)
        
        # Save summary
        with open(f"{output_path}/temporal_validation_summary.txt", 'w') as f:
            f.write("FCDT-TPFF Temporal Validation Summary\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Training set: {len(train_idx)} patients\n")
            f.write(f"Test set: {len(test_idx)} patients\n\n")
            
            f.write("Distribution Shift:\n")
            high_shift = report['distribution_shift'][
                report['distribution_shift']['psi'] > 0.25
            ]
            f.write(f"  Features with significant shift: {len(high_shift)}\n")
            
            if len(high_shift) > 0:
                f.write("\n  Top shifted features:\n")
                for _, row in high_shift.head().iterrows():
                    f.write(f"    - {row['feature']}: PSI = {row['psi']:.3f}\n")
        
        print("\n" + "="*80)
        print("TEMPORAL VALIDATION REPORT COMPLETE")
        print("="*80)
        print(f"Results saved to {output_path}/")
        
        return report
    
    def plot_temporal_shifts(self, data, train_idx, test_idx, output_path):
        """Visualize temporal distribution shifts"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Get time data
        train_times = pd.to_datetime(data.loc[train_idx, self.time_column])
        test_times = pd.to_datetime(data.loc[test_idx, self.time_column])
        
        # 1. Timeline
        ax = axes[0, 0]
        ax.hist([train_times, test_times], bins=20, label=['Training', 'Testing'], alpha=0.7)
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Temporal Distribution of Cohorts')
        ax.legend()
        
        # 2. Cumulative patients
        ax = axes[0, 1]
        all_times = pd.concat([train_times, test_times]).sort_values()
        all_times_reset = all_times.reset_index(drop=True)
        ax.plot(all_times_reset, np.arange(len(all_times_reset)))
        ax.axvline(train_times.max(), color='red', linestyle='--', label='Train/Test Split')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Patients')
        ax.set_title('Cumulative Patient Enrollment')
        ax.legend()
        
        # 3. Sample size over time
        ax = axes[1, 0]
        train_counts = train_times.dt.to_period('M').value_counts().sort_index()
        test_counts = test_times.dt.to_period('M').value_counts().sort_index()
        
        ax.bar(range(len(train_counts)), train_counts.values, label='Training', alpha=0.7)
        ax.bar(range(len(train_counts), len(train_counts) + len(test_counts)), 
               test_counts.values, label='Testing', alpha=0.7)
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Monthly Patient Distribution')
        ax.legend()
        
        # 4. Text summary
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
        Temporal Split Summary
        
        Training Set:
        • N = {len(train_idx)}
        • Date range: {train_times.min().date()} to {train_times.max().date()}
        • Duration: {(train_times.max() - train_times.min()).days} days
        
        Testing Set:
        • N = {len(test_idx)}
        • Date range: {test_times.min().date()} to {test_times.max().date()}
        • Duration: {(test_times.max() - test_times.min()).days} days
        
        Split Date: {train_times.max().date()}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/temporal_split_visualization.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Temporal visualization saved to {output_path}/temporal_split_visualization.png")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Load data
    demographics = pd.read_csv("./processed_data/demographics.csv")
    static_features = pd.read_csv("./processed_data/static_features.csv", index_col=0)
    
    # Initialize validator
    validator = TemporalValidator(time_column='admittime', random_state=42)
    
    # Generate comprehensive report
    report = validator.generate_temporal_validation_report(
        data=demographics,
        X=static_features,
        test_fraction=0.3,
        output_path="./processed_data"
    )
    
    print("\n✓ Temporal validation complete!")

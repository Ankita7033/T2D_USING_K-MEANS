"""
FCDT-TPFF: Statistical Rigor Framework for JBI Submission
✓ Bootstrap confidence intervals
✓ Permutation tests for cluster differences
✓ Multiple comparison correction
✓ Effect size reporting (Cohen's d, Cramér's V)
✓ Power analysis
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class StatisticalValidator:
    """
    Comprehensive statistical validation for clustering results
    """
    
    def __init__(self, alpha=0.05, n_bootstrap=1000, n_permutations=1000, random_state=42):
        self.alpha = alpha
        self.n_bootstrap = n_bootstrap
        self.n_permutations = n_permutations
        self.random_state = random_state
        np.random.seed(random_state)
    
    # =========================================================================
    # BOOTSTRAP CONFIDENCE INTERVALS
    # =========================================================================
    
    def bootstrap_clustering_metrics(self, embeddings, labels):
        """
        Calculate bootstrap CIs for clustering quality metrics
        
        Returns:
            dict: Metrics with 95% CIs
        """
        print("\n" + "="*60)
        print("BOOTSTRAP CONFIDENCE INTERVALS")
        print("="*60)
        
        metrics = {
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': []
        }
        
        n_samples = len(embeddings)
        
        for i in range(self.n_bootstrap):
            if i % 200 == 0:
                print(f"Bootstrap iteration {i}/{self.n_bootstrap}")
            
            # Resample with replacement
            indices = resample(np.arange(n_samples), replace=True, 
                             random_state=self.random_state + i)
            
            X_boot = embeddings[indices]
            y_boot = labels[indices]
            
            # Skip if all samples in same cluster
            if len(np.unique(y_boot)) < 2:
                continue
            
            try:
                metrics['silhouette'].append(silhouette_score(X_boot, y_boot))
                metrics['davies_bouldin'].append(davies_bouldin_score(X_boot, y_boot))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(X_boot, y_boot))
            except:
                continue
        
        results = {}
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            ci_lower = np.percentile(values, 2.5)
            ci_upper = np.percentile(values, 97.5)
            
            results[metric_name] = {
                'mean': mean_val,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': np.std(values)
            }
            
            print(f"\n{metric_name.upper()}:")
            print(f"  Mean: {mean_val:.4f}")
            print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"  Std: {np.std(values):.4f}")
        
        return results
    
    # =========================================================================
    # PERMUTATION TESTS
    # =========================================================================
    
    def permutation_test_cluster_differences(self, data, labels, feature_name):
        """
        Test if cluster differences in a feature are statistically significant
        
        Args:
            data: Feature values (array-like)
            labels: Cluster assignments
            feature_name: Name of feature being tested
            
        Returns:
            dict: Test results with p-value
        """
        # Check for zero variance
        if np.std(data) == 0 or len(np.unique(data)) == 1:
            return {
                'feature': feature_name,
                'h_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'note': 'zero_variance'
            }
        
        # Create cluster groups
        cluster_groups = [data[labels == k] for k in np.unique(labels)]
        
        # Check if any group has insufficient data
        if any(len(g) < 3 for g in cluster_groups):
            return {
                'feature': feature_name,
                'h_statistic': 0.0,
                'p_value': 1.0,
                'significant': False,
                'note': 'insufficient_data'
            }
        
        # Check if all groups are identical
        try:
            h_observed, _ = stats.kruskal(*cluster_groups)
        except ValueError as e:
            if 'identical' in str(e).lower():
                return {
                    'feature': feature_name,
                    'h_statistic': 0.0,
                    'p_value': 1.0,
                    'significant': False,
                    'note': 'identical_groups'
                }
            else:
                raise
        
        # Permutation distribution
        h_null = []
        for i in range(self.n_permutations):
            labels_perm = np.random.permutation(labels)
            cluster_groups_perm = [data[labels_perm == k] for k in np.unique(labels)]
            
            try:
                h_perm, _ = stats.kruskal(*cluster_groups_perm)
                h_null.append(h_perm)
            except ValueError:
                # If permutation creates identical groups, assign H=0
                h_null.append(0.0)
        
        # P-value
        p_value = np.mean(np.array(h_null) >= h_observed)
        
        return {
            'feature': feature_name,
            'h_statistic': h_observed,
            'p_value': p_value,
            'significant': p_value < self.alpha
        }

    def test_all_features(self, clinical_data, labels, feature_names):
        """
        Test all clinical features with multiple comparison correction
        
        Args:
            clinical_data: DataFrame with clinical features
            labels: Cluster assignments
            feature_names: List of features to test
            
        Returns:
            DataFrame: Results with corrected p-values
        """
        print("\n" + "="*60)
        print("PERMUTATION TESTS FOR CLUSTER DIFFERENCES")
        print("="*60)
        
        results = []
        skipped = []
        
        for feat in feature_names:
            if feat not in clinical_data.columns:
                continue
            
            data = clinical_data[feat].values
            
            # Skip if too many missing
            if np.isnan(data).sum() / len(data) > 0.5:
                skipped.append((feat, 'high_missingness'))
                continue
            
            # Remove NaN
            valid_idx = ~np.isnan(data)
            data_valid = data[valid_idx]
            labels_valid = labels[valid_idx]
            
            # Skip if insufficient data after removing NaN
            if len(data_valid) < 10:
                skipped.append((feat, 'insufficient_data'))
                continue
            
            result = self.permutation_test_cluster_differences(
                data_valid, labels_valid, feat
            )
            results.append(result)
        
        if not results:
            print("\n⚠ WARNING: No features could be tested!")
            print(f"Skipped {len(skipped)} features")
            return pd.DataFrame(columns=['feature', 'h_statistic', 'p_value', 
                                        'significant', 'p_corrected', 'significant_corrected'])
        
        results_df = pd.DataFrame(results)
        
        # Multiple comparison correction (Benjamini-Hochberg FDR)
        if len(results_df) > 0:
            _, p_corrected, _, _ = multipletests(
                results_df['p_value'].values,
                alpha=self.alpha,
                method='fdr_bh'
            )
            results_df['p_corrected'] = p_corrected
            results_df['significant_corrected'] = p_corrected < self.alpha
        
        print(f"\nTested {len(results_df)} features")
        if len(skipped) > 0:
            print(f"Skipped {len(skipped)} features:")
            for feat, reason in skipped[:5]:
                print(f"  - {feat}: {reason}")
            if len(skipped) > 5:
                print(f"  ... and {len(skipped) - 5} more")
        
        print(f"Significant (uncorrected): {results_df['significant'].sum()}")
        print(f"Significant (FDR-corrected): {results_df['significant_corrected'].sum()}")
        
        return results_df.sort_values('p_corrected')
        
    # =========================================================================
    # EFFECT SIZES
    # =========================================================================
    
    def cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d
    
    def cramers_v(self, confusion_matrix):
        """Calculate Cramér's V for categorical association"""
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum()
        min_dim = min(confusion_matrix.shape) - 1
        
        v = np.sqrt(chi2 / (n * min_dim))
        return v
    
    def calculate_effect_sizes(self, clinical_data, labels, feature_names):
        """
        Calculate effect sizes for all pairwise cluster comparisons
        
        Returns:
            DataFrame: Effect sizes for each feature
        """
        print("\n" + "="*60)
        print("EFFECT SIZE ANALYSIS")
        print("="*60)
        
        unique_clusters = np.unique(labels)
        results = []
        
        for feat in feature_names:
            if feat not in clinical_data.columns:
                continue
            
            data = clinical_data[feat].values
            valid_idx = ~np.isnan(data)
            data_valid = data[valid_idx]
            labels_valid = labels[valid_idx]
            
            # Pairwise comparisons
            for i, c1 in enumerate(unique_clusters):
                for c2 in unique_clusters[i+1:]:
                    g1 = data_valid[labels_valid == c1]
                    g2 = data_valid[labels_valid == c2]
                    
                    if len(g1) < 3 or len(g2) < 3:
                        continue
                    
                    d = self.cohens_d(g1, g2)
                    
                    # Interpret effect size
                    if abs(d) < 0.2:
                        magnitude = "negligible"
                    elif abs(d) < 0.5:
                        magnitude = "small"
                    elif abs(d) < 0.8:
                        magnitude = "medium"
                    else:
                        magnitude = "large"
                    
                    results.append({
                        'feature': feat,
                        'cluster1': int(c1),
                        'cluster2': int(c2),
                        'cohens_d': d,
                        'magnitude': magnitude
                    })
        
        results_df = pd.DataFrame(results)
        
        print(f"\nEffect sizes calculated for {len(feature_names)} features")
        print(f"Large effects (|d| > 0.8): {(results_df['cohens_d'].abs() > 0.8).sum()}")
        print(f"Medium effects (0.5 < |d| < 0.8): {((results_df['cohens_d'].abs() > 0.5) & (results_df['cohens_d'].abs() <= 0.8)).sum()}")
        
        return results_df
    
    # =========================================================================
    # STABILITY ANALYSIS
    # =========================================================================
    
    def stability_analysis(self, embeddings, n_clusters, n_iterations=100):
        """
        Assess clustering stability using adjusted Rand index
        
        Returns:
            dict: Stability metrics
        """
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        
        print("\n" + "="*60)
        print("CLUSTERING STABILITY ANALYSIS")
        print("="*60)
        
        # Reference clustering
        kmeans_ref = KMeans(n_clusters=n_clusters, n_init=30, random_state=self.random_state)
        labels_ref = kmeans_ref.fit_predict(embeddings)
        
        ari_scores = []
        
        for i in range(n_iterations):
            # Subsample 80% of data
            indices = resample(np.arange(len(embeddings)), 
                             n_samples=int(0.8*len(embeddings)),
                             replace=False,
                             random_state=self.random_state + i)
            
            X_sub = embeddings[indices]
            
            # Cluster subsample
            kmeans_sub = KMeans(n_clusters=n_clusters, n_init=10, 
                               random_state=self.random_state + i)
            labels_sub = kmeans_sub.fit_predict(X_sub)
            
            # Compare to reference (only overlapping samples)
            ari = adjusted_rand_score(labels_ref[indices], labels_sub)
            ari_scores.append(ari)
        
        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        
        # Interpretation
        if mean_ari > 0.8:
            stability = "excellent"
        elif mean_ari > 0.65:
            stability = "good"
        elif mean_ari > 0.5:
            stability = "moderate"
        else:
            stability = "poor"
        
        print(f"\nMean ARI: {mean_ari:.4f} ± {std_ari:.4f}")
        print(f"Stability: {stability}")
        
        return {
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            'min_ari': np.min(ari_scores),
            'max_ari': np.max(ari_scores),
            'stability': stability
        }
    
    # =========================================================================
    # POWER ANALYSIS
    # =========================================================================
    
    def post_hoc_power_analysis(self, clinical_data, labels, feature_name, alpha=0.05):
        """
        Calculate achieved statistical power for detecting cluster differences
        
        Returns:
            dict: Power analysis results
        """
        from statsmodels.stats.power import FTestAnovaPower
        
        data = clinical_data[feature_name].values
        valid_idx = ~np.isnan(data)
        data_valid = data[valid_idx]
        labels_valid = labels[valid_idx]
        
        # Calculate effect size (eta-squared)
        cluster_groups = [data_valid[labels_valid == k] for k in np.unique(labels_valid)]
        
        # ANOVA
        f_stat, _ = stats.f_oneway(*cluster_groups)
        
        # Convert F to effect size (Cohen's f)
        k = len(cluster_groups)  # number of groups
        n = len(data_valid)
        
        # Effect size f = sqrt(eta^2 / (1 - eta^2))
        # where eta^2 = SSB / SST
        grand_mean = np.mean(data_valid)
        ssb = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in cluster_groups)
        sst = np.sum((data_valid - grand_mean)**2)
        eta_squared = ssb / sst
        
        effect_size_f = np.sqrt(eta_squared / (1 - eta_squared))
        
        # Calculate power
        power_calc = FTestAnovaPower()
        power = power_calc.solve_power(
            effect_size=effect_size_f,
            nobs=n/k,  # observations per group
            alpha=alpha,
            k_groups=k
        )
        
        return {
            'feature': feature_name,
            'effect_size_f': effect_size_f,
            'eta_squared': eta_squared,
            'power': power,
            'n_total': n,
            'n_per_group': n/k,
            'adequate_power': power >= 0.8
        }
    
    # =========================================================================
    # COMPREHENSIVE REPORT
    # =========================================================================
    
    def generate_comprehensive_report(self, embeddings, labels, clinical_data, 
                                     feature_names, output_path="./processed_data"):
        """
        Generate complete statistical validation report
        
        This is the main function to call for JBI-ready statistics
        """
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE STATISTICAL REPORT")
        print("="*80)
        
        report = {}
        
        # 1. Bootstrap CIs for clustering metrics
        report['clustering_metrics'] = self.bootstrap_clustering_metrics(embeddings, labels)
        
        # 2. Permutation tests with multiple comparison correction
        report['feature_tests'] = self.test_all_features(clinical_data, labels, feature_names)
        
        # 3. Effect sizes
        report['effect_sizes'] = self.calculate_effect_sizes(clinical_data, labels, feature_names)
        
        # 4. Stability analysis
        report['stability'] = self.stability_analysis(embeddings, n_clusters=len(np.unique(labels)))
        
        # 5. Power analysis for top features
        significant_features = report['feature_tests'].head(5)['feature'].tolist()
        report['power_analysis'] = []
        
        print("\n" + "="*60)
        print("POST-HOC POWER ANALYSIS")
        print("="*60)
        
        for feat in significant_features:
            try:
                power_result = self.post_hoc_power_analysis(clinical_data, labels, feat)
                report['power_analysis'].append(power_result)
                
                print(f"\n{feat}:")
                print(f"  Effect size (f): {power_result['effect_size_f']:.4f}")
                print(f"  Power: {power_result['power']:.4f}")
                print(f"  Adequate (≥0.8): {power_result['adequate_power']}")
            except Exception as e:
                print(f"Could not calculate power for {feat}: {e}")
        
        # Save results
        report['feature_tests'].to_csv(f"{output_path}/statistical_tests.csv", index=False)
        report['effect_sizes'].to_csv(f"{output_path}/effect_sizes.csv", index=False)
        
        pd.DataFrame(report['power_analysis']).to_csv(
            f"{output_path}/power_analysis.csv", index=False
        )
        
        # Save summary
        summary = {
            'Clustering Metrics': report['clustering_metrics'],
            'Stability': report['stability'],
            'N Significant Features (FDR-corrected)': report['feature_tests']['significant_corrected'].sum(),
            'Features with Adequate Power': sum(p['adequate_power'] for p in report['power_analysis'])
        }
        
        with open(f"{output_path}/statistical_summary.txt", 'w') as f:
            f.write("FCDT-TPFF Statistical Validation Summary\n")
            f.write("="*60 + "\n\n")
            
            for key, value in summary.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v}\n")
                else:
                    f.write(f"{key}: {value}\n")
                f.write("\n")
        
        print("\n" + "="*80)
        print("STATISTICAL REPORT COMPLETE")
        print("="*80)
        print(f"Results saved to {output_path}/")
        
        return report


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Load your results
    embeddings = np.load("./processed_data/test_embeddings.npy")
    labels = np.load("./processed_data/test_labels.npy")
    
    # Load clinical data for validation
    clinical_data = pd.read_csv("./processed_data/static_features.csv", index_col=0)
    
    # Define features to test
    feature_names = clinical_data.columns.tolist()
    
    # Run comprehensive validation
    validator = StatisticalValidator(
        alpha=0.05,
        n_bootstrap=1000,
        n_permutations=1000,
        random_state=42
    )
    
    report = validator.generate_comprehensive_report(
        embeddings=embeddings,
        labels=labels,
        clinical_data=clinical_data,
        feature_names=feature_names
    )
    
    print("\n✓ Statistical validation complete!")
    print("Add these results to your manuscript's Results section.")

"""
Cluster Stability Analysis Module
Implements bootstrap and subsampling stability metrics for clustering validation

Required for journal acceptance - measures robustness of cluster assignments
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, jaccard_score
from sklearn.metrics.cluster import contingency_matrix
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class ClusterStabilityAnalyzer:
    """
    Comprehensive cluster stability analysis using:
    1. Bootstrap resampling
    2. Subsampling stability
    3. Per-cluster stability metrics
    """
    
    def __init__(self, n_iterations=100, subsample_ratio=0.8, random_state=42):
        """
        Args:
            n_iterations: Number of bootstrap/subsample iterations
            subsample_ratio: Fraction of data to use in each subsample (0.8 = 80%)
            random_state: Random seed for reproducibility
        """
        self.n_iterations = n_iterations
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Store results
        self.bootstrap_results = {}
        self.subsample_results = {}
        
    def bootstrap_stability(self, embeddings, n_clusters, original_labels=None):
        """
        Bootstrap stability: Resample patients WITH replacement
        
        Args:
            embeddings: Patient embeddings (n_patients, embedding_dim)
            n_clusters: Number of clusters
            original_labels: Original cluster labels (optional, for comparison)
            
        Returns:
            Dictionary with stability metrics
        """
        print(f"\n{'='*70}")
        print("BOOTSTRAP STABILITY ANALYSIS")
        print(f"{'='*70}")
        print(f"Iterations: {self.n_iterations}")
        print(f"Patients: {len(embeddings)}")
        print(f"Clusters: {n_clusters}")
        
        n_samples = len(embeddings)
        ari_scores = []
        jaccard_scores = []
        all_labels = []
        
        # Run bootstrap iterations
        for i in tqdm(range(self.n_iterations), desc="Bootstrap iterations"):
            # Resample WITH replacement
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            bootstrap_embeddings = embeddings[indices]
            
            # Cluster bootstrap sample
            kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.rng.randint(10000))
            bootstrap_labels = kmeans.fit_predict(bootstrap_embeddings)
            
            # Store labels (mapped back to original indices)
            full_labels = np.full(n_samples, -1)
            full_labels[indices] = bootstrap_labels
            all_labels.append(full_labels)
            
            # Compare consecutive runs
            if i > 0:
                # Only compare samples that appeared in both runs
                prev_labels = all_labels[i-1]
                valid_mask = (full_labels != -1) & (prev_labels != -1)
                
                if valid_mask.sum() > 0:
                    ari = adjusted_rand_score(
                        prev_labels[valid_mask], 
                        full_labels[valid_mask]
                    )
                    ari_scores.append(ari)
        
        # Compute statistics
        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        ci_lower = np.percentile(ari_scores, 2.5)
        ci_upper = np.percentile(ari_scores, 97.5)
        
        # Assess stability
        stability_level = self._assess_stability(mean_ari)
        
        results = {
            'method': 'bootstrap',
            'n_iterations': self.n_iterations,
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'median_ari': np.median(ari_scores),
            'all_ari_scores': ari_scores,
            'stability_level': stability_level,
            'all_labels': all_labels
        }
        
        self.bootstrap_results = results
        
        print(f"\n[OK] Bootstrap Stability Results:")
        print(f"  Mean ARI: {mean_ari:.4f} ± {std_ari:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Stability: {stability_level}")
        
        return results
    
    def subsample_stability(self, embeddings, n_clusters, original_labels=None):
        """
        Subsample stability: Use random subsets WITHOUT replacement
        More stringent test than bootstrap
        
        Args:
            embeddings: Patient embeddings
            n_clusters: Number of clusters
            original_labels: Original labels (optional)
            
        Returns:
            Dictionary with stability metrics
        """
        print(f"\n{'='*70}")
        print("SUBSAMPLE STABILITY ANALYSIS")
        print(f"{'='*70}")
        print(f"Iterations: {self.n_iterations}")
        print(f"Subsample ratio: {self.subsample_ratio} ({self.subsample_ratio*100:.0f}%)")
        
        n_samples = len(embeddings)
        subsample_size = int(n_samples * self.subsample_ratio)
        
        ari_scores = []
        jaccard_scores = []
        all_labels = []
        
        # Run subsample iterations
        for i in tqdm(range(self.n_iterations), desc="Subsample iterations"):
            # Random subsample WITHOUT replacement
            indices = self.rng.choice(n_samples, size=subsample_size, replace=False)
            subsample_embeddings = embeddings[indices]
            
            # Cluster subsample
            kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.rng.randint(10000))
            subsample_labels = kmeans.fit_predict(subsample_embeddings)
            
            # Store labels
            full_labels = np.full(n_samples, -1)
            full_labels[indices] = subsample_labels
            all_labels.append((indices, subsample_labels))
            
            # Compare consecutive runs on overlapping samples
            if i > 0:
                prev_indices, prev_labels_subset = all_labels[i-1]
                
                # Find overlapping samples
                overlap = np.intersect1d(indices, prev_indices)
                
                if len(overlap) > 10:  # Need sufficient overlap
                    # Get labels for overlapping samples
                    curr_mask = np.isin(indices, overlap)
                    prev_mask = np.isin(prev_indices, overlap)
                    
                    curr_overlap_labels = subsample_labels[curr_mask]
                    prev_overlap_labels = prev_labels_subset[prev_mask]
                    
                    ari = adjusted_rand_score(prev_overlap_labels, curr_overlap_labels)
                    ari_scores.append(ari)
        
        # Compute statistics
        mean_ari = np.mean(ari_scores)
        std_ari = np.std(ari_scores)
        ci_lower = np.percentile(ari_scores, 2.5)
        ci_upper = np.percentile(ari_scores, 97.5)
        
        stability_level = self._assess_stability(mean_ari)
        
        results = {
            'method': 'subsample',
            'n_iterations': self.n_iterations,
            'subsample_ratio': self.subsample_ratio,
            'mean_ari': mean_ari,
            'std_ari': std_ari,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'median_ari': np.median(ari_scores),
            'all_ari_scores': ari_scores,
            'stability_level': stability_level
        }
        
        self.subsample_results = results
        
        print(f"\n[OK] Subsample Stability Results:")
        print(f"  Mean ARI: {mean_ari:.4f} ± {std_ari:.4f}")
        print(f"  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  Stability: {stability_level}")
        
        return results
    
    def per_cluster_stability(self, embeddings, n_clusters, original_labels):
        """
        Compute stability for each individual cluster
        
        Args:
            embeddings: Patient embeddings
            n_clusters: Number of clusters
            original_labels: Reference cluster labels
            
        Returns:
            DataFrame with per-cluster stability metrics
        """
        print(f"\n{'='*70}")
        print("PER-CLUSTER STABILITY ANALYSIS")
        print(f"{'='*70}")
        
        n_samples = len(embeddings)
        cluster_stability = {i: [] for i in range(n_clusters)}
        
        # Run bootstrap iterations
        for i in tqdm(range(self.n_iterations), desc="Computing cluster stability"):
            indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            unique_indices = np.unique(indices)
            
            # Cluster bootstrap sample
            kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=self.rng.randint(10000))
            bootstrap_labels = kmeans.fit_predict(embeddings[unique_indices])
            
            # Map back to original indices
            label_map = dict(zip(unique_indices, bootstrap_labels))
            
            # For each cluster, compute Jaccard similarity
            for cluster_id in range(n_clusters):
                original_members = set(np.where(original_labels == cluster_id)[0])
                
                # Find corresponding cluster in bootstrap
                # Use majority vote for mapping
                original_in_bootstrap = [label_map.get(idx, -1) 
                                        for idx in original_members 
                                        if idx in label_map]
                
                if original_in_bootstrap:
                    bootstrap_cluster = max(set(original_in_bootstrap), 
                                          key=original_in_bootstrap.count)
                    bootstrap_members = set([idx for idx, lbl in label_map.items() 
                                           if lbl == bootstrap_cluster])
                    
                    # Jaccard similarity
                    intersection = len(original_members & bootstrap_members)
                    union = len(original_members | bootstrap_members)
                    jaccard = intersection / union if union > 0 else 0
                    
                    cluster_stability[cluster_id].append(jaccard)
        
        # Compile results
        stability_df = pd.DataFrame({
            'cluster': range(n_clusters),
            'mean_jaccard': [np.mean(cluster_stability[i]) for i in range(n_clusters)],
            'std_jaccard': [np.std(cluster_stability[i]) for i in range(n_clusters)],
            'ci_lower': [np.percentile(cluster_stability[i], 2.5) for i in range(n_clusters)],
            'ci_upper': [np.percentile(cluster_stability[i], 97.5) for i in range(n_clusters)],
            'cluster_size': [np.sum(original_labels == i) for i in range(n_clusters)]
        })
        
        stability_df['stability_level'] = stability_df['mean_jaccard'].apply(self._assess_stability)
        
        print("\n[OK] Per-Cluster Stability:")
        print(stability_df.to_string(index=False))
        
        return stability_df
    
    def _assess_stability(self, ari_score):
        """Interpret stability score"""
        if ari_score >= 0.8:
            return "Excellent"
        elif ari_score >= 0.6:
            return "Good"
        elif ari_score >= 0.4:
            return "Moderate"
        else:
            return "Poor"
    
    def generate_stability_report(self, output_path="./processed_data"):
        """
        Generate comprehensive stability report for manuscript
        
        Args:
            output_path: Directory to save report
        """
        print(f"\n{'='*70}")
        print("GENERATING STABILITY REPORT")
        print(f"{'='*70}")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CLUSTER STABILITY ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Bootstrap results
        if self.bootstrap_results:
            report_lines.append("1. BOOTSTRAP STABILITY (Resampling with Replacement)")
            report_lines.append("-" * 80)
            report_lines.append(f"Number of iterations: {self.bootstrap_results['n_iterations']}")
            report_lines.append(f"Mean ARI: {self.bootstrap_results['mean_ari']:.4f}")
            report_lines.append(f"Standard Deviation: {self.bootstrap_results['std_ari']:.4f}")
            report_lines.append(f"95% CI: [{self.bootstrap_results['ci_lower']:.4f}, {self.bootstrap_results['ci_upper']:.4f}]")
            report_lines.append(f"Median ARI: {self.bootstrap_results['median_ari']:.4f}")
            report_lines.append(f"Stability Assessment: {self.bootstrap_results['stability_level']}")
            report_lines.append("")
        
        # Subsample results
        if self.subsample_results:
            report_lines.append("2. SUBSAMPLE STABILITY (Random Subsets without Replacement)")
            report_lines.append("-" * 80)
            report_lines.append(f"Number of iterations: {self.subsample_results['n_iterations']}")
            report_lines.append(f"Subsample ratio: {self.subsample_results['subsample_ratio']}")
            report_lines.append(f"Mean ARI: {self.subsample_results['mean_ari']:.4f}")
            report_lines.append(f"Standard Deviation: {self.subsample_results['std_ari']:.4f}")
            report_lines.append(f"95% CI: [{self.subsample_results['ci_lower']:.4f}, {self.subsample_results['ci_upper']:.4f}]")
            report_lines.append(f"Median ARI: {self.subsample_results['median_ari']:.4f}")
            report_lines.append(f"Stability Assessment: {self.subsample_results['stability_level']}")
            report_lines.append("")
        
        # Interpretation
        report_lines.append("3. INTERPRETATION FOR MANUSCRIPT")
        report_lines.append("-" * 80)
        report_lines.append("ARI > 0.8: Excellent stability - clusters are highly reproducible")
        report_lines.append("ARI 0.6-0.8: Good stability - clusters are reliable")
        report_lines.append("ARI 0.4-0.6: Moderate stability - some variability in assignments")
        report_lines.append("ARI < 0.4: Poor stability - unstable clustering solution")
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report - FIX: Use UTF-8 encoding for Windows
        report_path = f"{output_path}/cluster_stability_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"[OK] Stability report saved to: {report_path}")
        
        return '\n'.join(report_lines)


def run_stability_analysis(embeddings, labels, n_clusters, output_path="./processed_data"):
    """
    Convenience function to run complete stability analysis
    
    Args:
        embeddings: Patient embeddings (numpy array)
        labels: Cluster labels (numpy array)
        n_clusters: Number of clusters
        output_path: Where to save results
        
    Returns:
        Dictionary with all stability results
    """
    analyzer = ClusterStabilityAnalyzer(n_iterations=100, subsample_ratio=0.8)
    
    # Run analyses
    bootstrap_results = analyzer.bootstrap_stability(embeddings, n_clusters, labels)
    subsample_results = analyzer.subsample_stability(embeddings, n_clusters, labels)
    cluster_results = analyzer.per_cluster_stability(embeddings, n_clusters, labels)
    
    # Generate report
    analyzer.generate_stability_report(output_path)
    
    # Save detailed results
    np.save(f"{output_path}/bootstrap_ari_scores.npy", bootstrap_results['all_ari_scores'])
    np.save(f"{output_path}/subsample_ari_scores.npy", subsample_results['all_ari_scores'])
    cluster_results.to_csv(f"{output_path}/per_cluster_stability.csv", index=False)
    
    print(f"\n{'='*70}")
    print("[OK] STABILITY ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print("\nSaved files:")
    print(f"  - {output_path}/cluster_stability_report.txt")
    print(f"  - {output_path}/bootstrap_ari_scores.npy")
    print(f"  - {output_path}/subsample_ari_scores.npy")
    print(f"  - {output_path}/per_cluster_stability.csv")
    
    return {
        'bootstrap': bootstrap_results,
        'subsample': subsample_results,
        'per_cluster': cluster_results,
        'analyzer': analyzer
    }

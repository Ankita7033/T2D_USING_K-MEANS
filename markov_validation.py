"""
Markov Transition Matrix Validation with Uncertainty Quantification
Critical for validating progression analysis

Implements:
1. Bootstrap validation of transition probabilities
2. Baseline comparisons (static persistence, random transitions)
3. Uncertainty quantification (95% CIs for all transitions)
"""

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class MarkovTransitionValidator:
    """
    Validates Markov transition matrices with uncertainty quantification
    """
    
    def __init__(self, n_bootstrap=1000, random_state=42):
        """
        Args:
            n_bootstrap: Number of bootstrap iterations
            random_state: Random seed
        """
        self.n_bootstrap = n_bootstrap
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        self.results = {}
        
    def compute_transition_matrix(self, trajectory_data):
        """
        Compute empirical transition matrix from longitudinal data
        
        Args:
            trajectory_data: DataFrame with columns ['patient_id', 'time', 'cluster']
            
        Returns:
            Transition matrix (n_clusters x n_clusters)
        """
        # Sort by patient and time
        df = trajectory_data.sort_values(['patient_id', 'time']).copy()
        
        # Get unique clusters
        clusters = sorted(df['cluster'].unique())
        n_clusters = len(clusters)
        
        # Initialize transition count matrix
        transition_counts = np.zeros((n_clusters, n_clusters))
        
        # Count transitions
        for patient_id in df['patient_id'].unique():
            patient_data = df[df['patient_id'] == patient_id]
            
            for i in range(len(patient_data) - 1):
                from_cluster = patient_data.iloc[i]['cluster']
                to_cluster = patient_data.iloc[i + 1]['cluster']
                
                transition_counts[from_cluster, to_cluster] += 1
        
        # Convert to probabilities (row-wise normalization)
        transition_matrix = np.zeros_like(transition_counts, dtype=float)
        for i in range(n_clusters):
            row_sum = transition_counts[i, :].sum()
            if row_sum > 0:
                transition_matrix[i, :] = transition_counts[i, :] / row_sum
        
        return transition_matrix, transition_counts
    
    def bootstrap_transitions(self, trajectory_data):
        """
        Bootstrap patient trajectories to get uncertainty estimates
        
        Args:
            trajectory_data: DataFrame with patient trajectories
            
        Returns:
            Dictionary with mean transitions and confidence intervals
        """
        print(f"\n{'='*70}")
        print("BOOTSTRAP TRANSITION MATRIX VALIDATION")
        print(f"{'='*70}")
        print(f"Bootstrap iterations: {self.n_bootstrap}")
        
        patients = trajectory_data['patient_id'].unique()
        n_patients = len(patients)
        n_clusters = len(trajectory_data['cluster'].unique())
        
        # Store transition matrices from each bootstrap
        all_transitions = []
        
        for i in tqdm(range(self.n_bootstrap), desc="Bootstrap iterations"):
            # Resample patients WITH replacement
            sampled_patients = self.rng.choice(patients, size=n_patients, replace=True)
            
            # Get data for sampled patients
            bootstrap_data = trajectory_data[
                trajectory_data['patient_id'].isin(sampled_patients)
            ].copy()
            
            # Compute transition matrix
            trans_matrix, _ = self.compute_transition_matrix(bootstrap_data)
            all_transitions.append(trans_matrix)
        
        # Convert to numpy array
        all_transitions = np.array(all_transitions)  # Shape: (n_bootstrap, n_clusters, n_clusters)
        
        # Compute statistics
        mean_transitions = np.mean(all_transitions, axis=0)
        std_transitions = np.std(all_transitions, axis=0)
        ci_lower = np.percentile(all_transitions, 2.5, axis=0)
        ci_upper = np.percentile(all_transitions, 97.5, axis=0)
        
        results = {
            'mean': mean_transitions,
            'std': std_transitions,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'all_bootstrap': all_transitions
        }
        
        self.results['bootstrap'] = results
        
        print("\n[OK] Bootstrap validation complete")
        print(f"Mean transition matrix:")
        print(mean_transitions)
        
        return results
    
    def static_persistence_baseline(self, trajectory_data):
        """
        Baseline 1: Static persistence (patients never transition)
        
        Args:
            trajectory_data: Patient trajectories
            
        Returns:
            Static persistence transition matrix (identity matrix)
        """
        print("\n" + "="*70)
        print("BASELINE 1: Static Persistence (No Transitions)")
        print("="*70)
        
        n_clusters = len(trajectory_data['cluster'].unique())
        
        # Identity matrix = 100% stay in same cluster
        static_matrix = np.eye(n_clusters)
        
        print("Static persistence matrix:")
        print(static_matrix)
        
        self.results['static_baseline'] = static_matrix
        
        return static_matrix
    
    def random_transition_baseline(self, trajectory_data):
        """
        Baseline 2: Random transitions (uniform probability to all clusters)
        
        Args:
            trajectory_data: Patient trajectories
            
        Returns:
            Random transition matrix
        """
        print("\n" + "="*70)
        print("BASELINE 2: Random Transitions (Uniform Distribution)")
        print("="*70)
        
        n_clusters = len(trajectory_data['cluster'].unique())
        
        # Uniform probability to all clusters
        random_matrix = np.ones((n_clusters, n_clusters)) / n_clusters
        
        print("Random transition matrix:")
        print(random_matrix)
        
        self.results['random_baseline'] = random_matrix
        
        return random_matrix
    
    def compare_to_baselines(self, trajectory_data):
        """
        Compare empirical transitions to baselines
        
        Args:
            trajectory_data: Patient trajectories
            
        Returns:
            Comparison statistics
        """
        print("\n" + "="*70)
        print("COMPARING TO BASELINES")
        print("="*70)
        
        # Get empirical and bootstrap results
        empirical_matrix, _ = self.compute_transition_matrix(trajectory_data)
        bootstrap_results = self.results.get('bootstrap', 
                                            self.bootstrap_transitions(trajectory_data))
        mean_matrix = bootstrap_results['mean']
        
        # Get baselines
        static_baseline = self.static_persistence_baseline(trajectory_data)
        random_baseline = self.random_transition_baseline(trajectory_data)
        
        # Compute distances (Frobenius norm)
        dist_to_static = np.linalg.norm(mean_matrix - static_baseline, 'fro')
        dist_to_random = np.linalg.norm(mean_matrix - random_baseline, 'fro')
        
        # Diagonal dominance (measure of stability)
        diagonal_sum = np.trace(mean_matrix)
        off_diagonal_sum = np.sum(mean_matrix) - diagonal_sum
        
        comparison = {
            'empirical_matrix': empirical_matrix,
            'mean_bootstrap_matrix': mean_matrix,
            'distance_to_static': dist_to_static,
            'distance_to_random': dist_to_random,
            'diagonal_dominance': diagonal_sum / mean_matrix.shape[0],
            'transition_rate': off_diagonal_sum / mean_matrix.size
        }
        
        print(f"\nDistance to static persistence: {dist_to_static:.4f}")
        print(f"Distance to random transitions: {dist_to_random:.4f}")
        print(f"Diagonal dominance: {comparison['diagonal_dominance']:.4f}")
        print(f"Transition rate: {comparison['transition_rate']:.4f}")
        
        # Interpretation
        print("\nInterpretation:")
        if dist_to_static < dist_to_random:
            print("  → Transitions show more stability (closer to static)")
        else:
            print("  → Transitions show more variability (closer to random)")
        
        if comparison['diagonal_dominance'] > 0.7:
            print("  → High diagonal dominance: Most patients remain stable")
        else:
            print("  → Low diagonal dominance: Frequent cluster transitions")
        
        self.results['comparison'] = comparison
        
        return comparison
    
    def statistical_significance_test(self, trajectory_data):
        """
        Test if transitions differ significantly from baselines
        
        Uses permutation test
        
        Args:
            trajectory_data: Patient trajectories
            
        Returns:
            p-values for comparisons
        """
        print("\n" + "="*70)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*70)
        
        empirical_matrix, _ = self.compute_transition_matrix(trajectory_data)
        static_baseline = self.results['static_baseline']
        random_baseline = self.results['random_baseline']
        
        # Compute test statistic (Frobenius distance)
        obs_dist_static = np.linalg.norm(empirical_matrix - static_baseline, 'fro')
        obs_dist_random = np.linalg.norm(empirical_matrix - random_baseline, 'fro')
        
        # Permutation test
        n_permutations = 1000
        perm_dist_static = []
        perm_dist_random = []
        
        print(f"\nRunning permutation tests ({n_permutations} iterations)...")
        
        for i in tqdm(range(n_permutations)):
            # Shuffle cluster labels within each time point
            perm_data = trajectory_data.copy()
            
            for time in perm_data['time'].unique():
                time_mask = perm_data['time'] == time
                perm_data.loc[time_mask, 'cluster'] = \
                    self.rng.permutation(perm_data.loc[time_mask, 'cluster'].values)
            
            # Compute permuted transition matrix
            perm_matrix, _ = self.compute_transition_matrix(perm_data)
            
            perm_dist_static.append(np.linalg.norm(perm_matrix - static_baseline, 'fro'))
            perm_dist_random.append(np.linalg.norm(perm_matrix - random_baseline, 'fro'))
        
        # Compute p-values
        p_value_static = np.mean(np.array(perm_dist_static) >= obs_dist_static)
        p_value_random = np.mean(np.array(perm_dist_random) <= obs_dist_random)
        
        print(f"\nResults:")
        print(f"  p-value (different from static): {p_value_static:.4f}")
        print(f"  p-value (different from random): {p_value_random:.4f}")
        
        significance = {
            'p_value_vs_static': p_value_static,
            'p_value_vs_random': p_value_random,
            'significant_vs_static': p_value_static < 0.05,
            'significant_vs_random': p_value_random < 0.05
        }
        
        self.results['significance'] = significance
        
        return significance
    
    def generate_transition_report(self, output_path="./processed_data"):
        """
        Generate comprehensive transition analysis report
        
        Args:
            output_path: Where to save report
        """
        print(f"\n{'='*70}")
        print("GENERATING TRANSITION VALIDATION REPORT")
        print(f"{'='*70}")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MARKOV TRANSITION MATRIX VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Bootstrap results
        if 'bootstrap' in self.results:
            bootstrap = self.results['bootstrap']
            report_lines.append("1. TRANSITION PROBABILITIES (Bootstrap Validated)")
            report_lines.append("-" * 80)
            report_lines.append("\nMean Transition Matrix:")
            report_lines.append(str(bootstrap['mean']))
            report_lines.append("\nStandard Deviations:")
            report_lines.append(str(bootstrap['std']))
            report_lines.append("")
        
        # Baseline comparison
        if 'comparison' in self.results:
            comp = self.results['comparison']
            report_lines.append("2. BASELINE COMPARISONS")
            report_lines.append("-" * 80)
            report_lines.append(f"Distance to static persistence: {comp['distance_to_static']:.4f}")
            report_lines.append(f"Distance to random transitions: {comp['distance_to_random']:.4f}")
            report_lines.append(f"Diagonal dominance: {comp['diagonal_dominance']:.4f}")
            report_lines.append(f"Overall transition rate: {comp['transition_rate']:.4f}")
            report_lines.append("")
        
        # Statistical significance
        if 'significance' in self.results:
            sig = self.results['significance']
            report_lines.append("3. STATISTICAL SIGNIFICANCE")
            report_lines.append("-" * 80)
            report_lines.append(f"p-value (vs static): {sig['p_value_vs_static']:.4f}")
            report_lines.append(f"p-value (vs random): {sig['p_value_vs_random']:.4f}")
            
            if sig['significant_vs_static']:
                report_lines.append("[PASS] Significantly different from static persistence (p<0.05)")
            if sig['significant_vs_random']:
                report_lines.append("[PASS] Significantly different from random transitions (p<0.05)")
            report_lines.append("")
        
        # Interpretation
        report_lines.append("4. INTERPRETATION FOR MANUSCRIPT")
        report_lines.append("-" * 80)
        report_lines.append("The transition matrix shows:")
        
        if 'comparison' in self.results:
            if self.results['comparison']['diagonal_dominance'] > 0.7:
                report_lines.append("- High stability: Most patients remain in their assigned cluster")
            else:
                report_lines.append("- Moderate transitions: Patients show progression patterns")
        
        if 'significance' in self.results:
            if self.results['significance']['significant_vs_static']:
                report_lines.append("- Transitions are non-trivial (not just measurement noise)")
            if self.results['significance']['significant_vs_random']:
                report_lines.append("- Transitions follow structured patterns (not random)")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report - FIX: Use UTF-8 encoding for Windows
        report_path = f"{output_path}/markov_validation_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"[OK] Report saved to: {report_path}")
        
        return '\n'.join(report_lines)
    
    def plot_transition_heatmap(self, output_path="./processed_data"):
        """
        Visualize transition matrix with confidence intervals
        
        Args:
            output_path: Where to save figure
        """
        if 'bootstrap' not in self.results:
            print("[WARNING] No bootstrap results available. Run bootstrap_transitions first.")
            return
        
        bootstrap = self.results['bootstrap']
        mean_matrix = bootstrap['mean']
        ci_lower = bootstrap['ci_lower']
        ci_upper = bootstrap['ci_upper']
        
        n_clusters = mean_matrix.shape[0]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Mean transitions
        sns.heatmap(mean_matrix, annot=True, fmt='.3f', cmap='Blues',
                   vmin=0, vmax=1, cbar_kws={'label': 'Transition Probability'},
                   ax=axes[0], linewidths=0.5, linecolor='gray')
        axes[0].set_xlabel('To Cluster', fontsize=12)
        axes[0].set_ylabel('From Cluster', fontsize=12)
        axes[0].set_title('Mean Transition Probabilities\n(Bootstrap Validated)', 
                         fontsize=13, fontweight='bold')
        
        # Plot 2: Confidence interval widths
        ci_width = ci_upper - ci_lower
        sns.heatmap(ci_width, annot=True, fmt='.3f', cmap='Reds',
                   cbar_kws={'label': '95% CI Width'},
                   ax=axes[1], linewidths=0.5, linecolor='gray')
        axes[1].set_xlabel('To Cluster', fontsize=12)
        axes[1].set_ylabel('From Cluster', fontsize=12)
        axes[1].set_title('Uncertainty (95% CI Width)', 
                         fontsize=13, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        plt.savefig(f"{output_path}/transition_matrix_heatmap.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{output_path}/transition_matrix_heatmap.pdf", bbox_inches='tight')
        plt.close()
        
        print(f"[OK] Heatmap saved to: {output_path}/transition_matrix_heatmap.png/pdf")


def run_markov_validation(trajectory_data, output_path="./processed_data"):
    """
    Convenience function to run complete Markov validation
    
    Args:
        trajectory_data: DataFrame with ['patient_id', 'time', 'cluster']
        output_path: Where to save results
        
    Returns:
        Validation results
    """
    validator = MarkovTransitionValidator(n_bootstrap=1000)
    
    # Run analyses
    validator.bootstrap_transitions(trajectory_data)
    validator.compare_to_baselines(trajectory_data)
    validator.statistical_significance_test(trajectory_data)
    
    # Generate outputs
    validator.generate_transition_report(output_path)
    validator.plot_transition_heatmap(output_path)
    
    # Save numerical results
    if 'bootstrap' in validator.results:
        np.save(f"{output_path}/transition_mean.npy", 
                validator.results['bootstrap']['mean'])
        np.save(f"{output_path}/transition_ci_lower.npy", 
                validator.results['bootstrap']['ci_lower'])
        np.save(f"{output_path}/transition_ci_upper.npy", 
                validator.results['bootstrap']['ci_upper'])
    
    print(f"\n{'='*70}")
    print("[OK] MARKOV VALIDATION COMPLETE")
    print(f"{'='*70}")
    print("\nSaved files:")
    print(f"  - {output_path}/markov_validation_report.txt")
    print(f"  - {output_path}/transition_matrix_heatmap.png/pdf")
    print(f"  - {output_path}/transition_mean.npy")
    print(f"  - {output_path}/transition_ci_lower.npy")
    print(f"  - {output_path}/transition_ci_upper.npy")
    
    return validator.results

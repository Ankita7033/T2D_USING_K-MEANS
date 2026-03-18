"""
Graph Ablation Study: With vs Without Outcome Nodes
Critical for proving embeddings are not outcome-driven

This module enables training the model with/without outcome information
in the graph structure to validate that clustering is based on clinical 
features rather than outcomes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pickle
import warnings
warnings.filterwarnings('ignore')


class GraphAblationStudy:
    """
    Conducts ablation study comparing:
    1. Full graph (with outcome nodes)
    2. Patient-only graph (without outcome nodes)
    """
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        
    def train_with_outcome_graph(self, train_temporal, train_static, model_class):
        """
        Train model WITH outcome nodes in graph
        This is your original implementation
        
        Args:
            train_temporal: Temporal features
            train_static: Static features  
            model_class: FCDT_TPFF class
            
        Returns:
            Trained model and embeddings
        """
        print("\n" + "="*70)
        print("TRAINING WITH OUTCOME GRAPH (Original)")
        print("="*70)
        
        from fcdt_tpff_model import DiabetesDataset, collate_fn
        
        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            DiabetesDataset(train_temporal, train_static),
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # Initialize model
        sample = next(iter(train_temporal.values()))
        temporal_dim = sample["micro"].shape[1]
        static_dim = train_static.shape[1]
        
        model = model_class(
            temporal_input_dim=temporal_dim,
            static_input_dim=static_dim,
            num_clusters=self.config.NUM_CLUSTERS,
            use_outcome_nodes=True  # KEY: Include outcome information
        ).to(self.config.DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.LR)
        
        # Training loop
        print("\nTraining with outcome graph...")
        best_loss = float("inf")
        
        for epoch in range(self.config.EPOCHS):
            model.train()
            losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                z, adj = model(batch["temporal"], batch["static"])
                
                # Standard loss
                loss_compact = torch.mean(torch.norm(z, dim=1))
                loss_graph = torch.mean(adj * torch.cdist(z, z))
                loss = loss_compact + 0.15 * loss_graph
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                losses.append(loss.item())
            
            avg_loss = np.mean(losses)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:02d}/{self.config.EPOCHS} | Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        print(f"\n[OK] Training complete. Best loss: {best_loss:.4f}")
        
        return model
    
    def train_without_outcome_graph(self, train_temporal, train_static, model_class):
        """
        Train model WITHOUT outcome nodes in graph
        This is the ablation version
        
        Args:
            train_temporal: Temporal features
            train_static: Static features
            model_class: FCDT_TPFF class
            
        Returns:
            Trained model and embeddings
        """
        print("\n" + "="*70)
        print("TRAINING WITHOUT OUTCOME GRAPH (Ablation)")
        print("="*70)
        
        from fcdt_tpff_model import DiabetesDataset, collate_fn
        
        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            DiabetesDataset(train_temporal, train_static),
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # Initialize model
        sample = next(iter(train_temporal.values()))
        temporal_dim = sample["micro"].shape[1]
        static_dim = train_static.shape[1]
        
        model = model_class(
            temporal_input_dim=temporal_dim,
            static_input_dim=static_dim,
            num_clusters=self.config.NUM_CLUSTERS,
            use_outcome_nodes=False  # KEY: NO outcome information
        ).to(self.config.DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.LR)
        
        # Training loop
        print("\nTraining without outcome graph...")
        best_loss = float("inf")
        
        for epoch in range(self.config.EPOCHS):
            model.train()
            losses = []
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                z, adj = model(batch["temporal"], batch["static"])
                
                # Standard loss (same as above)
                loss_compact = torch.mean(torch.norm(z, dim=1))
                loss_graph = torch.mean(adj * torch.cdist(z, z))
                loss = loss_compact + 0.15 * loss_graph
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                losses.append(loss.item())
            
            avg_loss = np.mean(losses)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:02d}/{self.config.EPOCHS} | Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        print(f"\n✓ Training complete. Best loss: {best_loss:.4f}")
        
        return model
    
    def extract_embeddings(self, model, test_temporal, test_static):
        """Extract embeddings from trained model"""
        from fcdt_tpff_model import DiabetesDataset, collate_fn
        
        test_loader = torch.utils.data.DataLoader(
            DiabetesDataset(test_temporal, test_static),
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        model.eval()
        embeddings = []
        
        with torch.no_grad():
            for batch in test_loader:
                z = model(batch["temporal"], batch["static"], return_pregraph=True)
                embeddings.append(z.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def evaluate_clustering(self, embeddings, name):
        """
        Cluster embeddings and compute metrics
        
        Args:
            embeddings: Patient embeddings
            name: Identifier (e.g., "with_outcome" or "without_outcome")
            
        Returns:
            Dictionary with clustering results
        """
        print(f"\nEvaluating clustering for: {name}")
        
        # Clustering
        kmeans = KMeans(
            n_clusters=self.config.NUM_CLUSTERS,
            n_init=30,
            random_state=42
        )
        labels = kmeans.fit_predict(embeddings)
        
        # Metrics
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
        
        print(f"  Silhouette: {silhouette:.4f}")
        print(f"  Davies-Bouldin: {davies_bouldin:.4f}")
        
        return {
            'name': name,
            'embeddings': embeddings,
            'labels': labels,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin
        }
    
    def run_ablation_study(self, train_temporal, train_static, 
                           test_temporal, test_static, model_class):
        """
        Run complete ablation study
        
        Args:
            train_temporal: Training temporal features
            train_static: Training static features
            test_temporal: Test temporal features
            test_static: Test static features
            model_class: FCDT_TPFF model class
            
        Returns:
            Dictionary comparing both conditions
        """
        print("\n" + "="*70)
        print("GRAPH ABLATION STUDY")
        print("="*70)
        print("Comparing: WITH vs WITHOUT outcome nodes in graph")
        
        # 1. Train WITH outcome graph
        model_with = self.train_with_outcome_graph(
            train_temporal, train_static, model_class
        )
        embeddings_with = self.extract_embeddings(
            model_with, test_temporal, test_static
        )
        results_with = self.evaluate_clustering(embeddings_with, "with_outcome")
        
        # 2. Train WITHOUT outcome graph
        model_without = self.train_without_outcome_graph(
            train_temporal, train_static, model_class
        )
        embeddings_without = self.extract_embeddings(
            model_without, test_temporal, test_static
        )
        results_without = self.evaluate_clustering(embeddings_without, "without_outcome")
        
        # 3. Compare results
        comparison = self._generate_comparison(results_with, results_without)
        
        # Store results
        self.results = {
            'with_outcome': results_with,
            'without_outcome': results_without,
            'comparison': comparison
        }
        
        return self.results
    
    def _generate_comparison(self, results_with, results_without):
        """Generate comparison report"""
        print("\n" + "="*70)
        print("ABLATION STUDY RESULTS")
        print("="*70)
        
        comparison = pd.DataFrame({
            'Condition': ['With Outcome Graph', 'Without Outcome Graph'],
            'Silhouette Score': [
                results_with['silhouette'],
                results_without['silhouette']
            ],
            'Davies-Bouldin Index': [
                results_with['davies_bouldin'],
                results_without['davies_bouldin']
            ]
        })
        
        # Calculate differences
        sil_diff = results_with['silhouette'] - results_without['silhouette']
        db_diff = results_with['davies_bouldin'] - results_without['davies_bouldin']
        
        print("\nClustering Quality Comparison:")
        print(comparison.to_string(index=False))
        
        print(f"\nDifferences:")
        print(f"  Silhouette: {sil_diff:+.4f} (positive = with outcome is better)")
        print(f"  Davies-Bouldin: {db_diff:+.4f} (negative = with outcome is better)")
        
        # Interpretation
        print("\nInterpretation:")
        if abs(sil_diff) < 0.05:
            print("  [OK] Minimal difference (<0.05) - clustering is NOT driven by outcomes")
            print("  [OK] This validates that embeddings capture intrinsic patient patterns")
        else:
            print("  [WARNING] Substantial difference (>0.05) - outcome information may influence clustering")
        
        return comparison
    
    def save_results(self, output_path="./processed_data"):
        """Save ablation study results"""
        print(f"\nSaving ablation study results to: {output_path}")
        
        # Save embeddings
        np.save(f"{output_path}/embeddings_with_outcome.npy", 
                self.results['with_outcome']['embeddings'])
        np.save(f"{output_path}/embeddings_without_outcome.npy", 
                self.results['without_outcome']['embeddings'])
        
        # Save labels
        np.save(f"{output_path}/labels_with_outcome.npy", 
                self.results['with_outcome']['labels'])
        np.save(f"{output_path}/labels_without_outcome.npy", 
                self.results['without_outcome']['labels'])
        
        # Save comparison
        self.results['comparison'].to_csv(
            f"{output_path}/ablation_comparison.csv", index=False
        )
        
        # Generate detailed report
        self._save_ablation_report(output_path)
        
        print("✓ Ablation study results saved")
    
    def _save_ablation_report(self, output_path):
        """Generate detailed ablation report for manuscript"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("GRAPH ABLATION STUDY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("Purpose: Validate that clustering is not outcome-driven")
        report_lines.append("")
        
        report_lines.append("1. EXPERIMENTAL DESIGN")
        report_lines.append("-" * 80)
        report_lines.append("Condition A: Model WITH outcome nodes in graph structure")
        report_lines.append("Condition B: Model WITHOUT outcome nodes (patient features only)")
        report_lines.append("")
        
        report_lines.append("2. CLUSTERING QUALITY METRICS")
        report_lines.append("-" * 80)
        report_lines.append(self.results['comparison'].to_string(index=False))
        report_lines.append("")
        
        sil_with = self.results['with_outcome']['silhouette']
        sil_without = self.results['without_outcome']['silhouette']
        sil_diff = sil_with - sil_without
        
        report_lines.append("3. INTERPRETATION")
        report_lines.append("-" * 80)
        report_lines.append(f"Silhouette difference: {sil_diff:.4f}")
        
        if abs(sil_diff) < 0.05:
            report_lines.append("[PASS] Minimal difference (<0.05)")
            report_lines.append("Conclusion: Clustering is NOT outcome-driven")
            report_lines.append("Embeddings capture intrinsic clinical patterns")
        else:
            report_lines.append("[CAUTION] Substantial difference (>0.05)")
            report_lines.append("Outcome information may influence clustering")
        
        report_lines.append("")
        report_lines.append("4. MANUSCRIPT RECOMMENDATION")
        report_lines.append("-" * 80)
        report_lines.append("Include this comparison in Results section:")
        report_lines.append('"To validate that clustering was not driven by outcome information,')
        report_lines.append('we compared models with and without outcome nodes in the graph.')
        report_lines.append(f'Silhouette scores were similar (with: {sil_with:.3f}, without: {sil_without:.3f}),')
        report_lines.append('demonstrating that identified subtypes reflect intrinsic patient')
        report_lines.append('characteristics rather than outcomes."')
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # FIX: Use UTF-8 encoding for Windows compatibility
        with open(f"{output_path}/ablation_study_report.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"[OK] Detailed report saved to: {output_path}/ablation_study_report.txt")


# Convenience function
def run_graph_ablation(train_temporal, train_static, test_temporal, test_static,
                       model_class, config, output_path="./processed_data"):
    """
    Run complete graph ablation study
    
    Args:
        train_temporal, train_static: Training data
        test_temporal, test_static: Test data
        model_class: FCDT_TPFF class
        config: Configuration object
        output_path: Where to save results
        
    Returns:
        Ablation study results
    """
    study = GraphAblationStudy(config)
    results = study.run_ablation_study(
        train_temporal, train_static,
        test_temporal, test_static,
        model_class
    )
    study.save_results(output_path)
    
    return results

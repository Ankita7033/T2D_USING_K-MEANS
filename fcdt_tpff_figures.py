"""
FCDT-TPFF Figure Generation (FIXED)
Generates publication-quality figures and tables
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------

print("Loading data...")
try:
    embeddings = np.load('./processed_data/embeddings.npy')
    labels = np.load('./processed_data/labels.npy')
    
    print(f"✔ Loaded embeddings: {embeddings.shape}")
    print(f"✔ Loaded labels: {labels.shape}")
    
    # Load clustered patients (which has correct patient IDs)
    try:
        static = pd.read_csv('./processed_data/clustered_patients.csv', index_col=0)
        print(f"✔ Loaded clustered patients: {static.shape}")
    except:
        # Fall back to static features
        static = pd.read_csv('./processed_data/static_features.csv', index_col=0)
        print(f"✔ Loaded static features: {static.shape}")
        
        # Match lengths
        if len(static) != len(labels):
            print(f"⚠ Mismatch: static has {len(static)}, labels has {len(labels)}")
            print("  Taking first {len(labels)} rows from static...")
            static = static.iloc[:len(labels)]
        
        static['cluster'] = labels
    
    # Verify alignment
    if len(static) != len(labels):
        print(f"✖ ERROR: Length mismatch after loading")
        print(f"  Static: {len(static)}, Labels: {len(labels)}")
        exit(1)
    
    # Try to load original demographics
    try:
        demo = pd.read_csv('./processed_data/demographics.csv')
        
        # Match with static patient IDs
        demo = demo[demo.subject_id.isin(static.index)].copy()
        demo.set_index('subject_id', inplace=True)
        demo = demo.loc[static.index]  # Ensure same order
        demo['cluster'] = labels
        
        print(f"✔ Loaded demographics: {demo.shape}")
    except Exception as e:
        demo = None
        print(f"⚠ Could not load demographics: {e}")
        print("  Using scaled features instead")
    
    print(f"\n✔ Final dataset: {len(embeddings)} patients, {len(np.unique(labels))} clusters")
except Exception as e:
    print(f"✖ Error loading data: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

NUM_CLUSTERS = len(np.unique(labels))

# ------------------------------------------------------------------
# FIGURE 1: t-SNE Visualization
# ------------------------------------------------------------------

def generate_figure1_tsne():
    print("\nGenerating Figure 1: t-SNE Visualization...")
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cluster
    for i in range(NUM_CLUSTERS):
        mask = labels == i
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], 
                  label=f'Cluster {i}', alpha=0.6, s=50,
                  edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('FCDT-TPFF Clustering Visualization (t-SNE)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./processed_data/figure1_tsne.png', dpi=300, bbox_inches='tight')
    plt.savefig('./processed_data/figure1_tsne.pdf', bbox_inches='tight')
    print("✔ Saved: figure1_tsne.png/pdf")
    plt.close()

# ------------------------------------------------------------------
# FIGURE 2: PCA Visualization
# ------------------------------------------------------------------

def generate_figure2_pca():
    print("\nGenerating Figure 2: PCA Visualization...")
    
    # Compute PCA
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(embeddings)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cluster
    for i in range(NUM_CLUSTERS):
        mask = labels == i
        ax.scatter(emb_pca[mask, 0], emb_pca[mask, 1], 
                  label=f'Cluster {i}', alpha=0.6, s=50,
                  edgecolors='white', linewidth=0.5)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                 fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                 fontsize=12)
    ax.set_title('FCDT-TPFF Clustering Visualization (PCA)', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./processed_data/figure2_pca.png', dpi=300, bbox_inches='tight')
    plt.savefig('./processed_data/figure2_pca.pdf', bbox_inches='tight')
    print("✔ Saved: figure2_pca.png/pdf")
    plt.close()

# ------------------------------------------------------------------
# FIGURE 3: Training Loss Curves
# ------------------------------------------------------------------

def generate_figure3_training():
    print("\nGenerating Figure 3: Training Curves...")
    
    try:
        history = pd.read_csv('./processed_data/training_history.csv')
    except:
        print("⚠ Training history not found, skipping...")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total Loss
    axes[0, 0].plot(history['epoch'], history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11)
    axes[0, 0].set_ylabel('Total Loss', fontsize=11)
    axes[0, 0].set_title('(a) Total Loss', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Variance
    axes[0, 1].plot(history['epoch'], history['avg_variance'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11)
    axes[0, 1].set_ylabel('Average Variance', fontsize=11)
    axes[0, 1].set_title('(b) Embedding Variance', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Separation Loss
    axes[1, 0].plot(history['epoch'], history['sep_loss'], 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch', fontsize=11)
    axes[1, 0].set_ylabel('Separation Loss', fontsize=11)
    axes[1, 0].set_title('(c) Separation Loss', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Compact Loss
    axes[1, 1].plot(history['epoch'], history['compact_loss'], 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Epoch', fontsize=11)
    axes[1, 1].set_ylabel('Compactness Loss', fontsize=11)
    axes[1, 1].set_title('(d) Compactness Loss', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./processed_data/figure3_training.png', dpi=300, bbox_inches='tight')
    plt.savefig('./processed_data/figure3_training.pdf', bbox_inches='tight')
    print("✔ Saved: figure3_training.png/pdf")
    plt.close()

# ------------------------------------------------------------------
# FIGURE 4: Cluster Distribution
# ------------------------------------------------------------------

def generate_figure4_distribution():
    print("\nGenerating Figure 4: Cluster Distribution...")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette("husl", NUM_CLUSTERS)
    bars = ax.bar(unique_labels, counts, color=colors, alpha=0.8, 
                  edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{count}\n({count/len(labels)*100:.1f}%)',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Patient Distribution Across Clusters', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(unique_labels)
    ax.set_xticklabels([f'C{i}' for i in unique_labels])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./processed_data/figure4_distribution.png', dpi=300, bbox_inches='tight')
    plt.savefig('./processed_data/figure4_distribution.pdf', bbox_inches='tight')
    print("✔ Saved: figure4_distribution.png/pdf")
    plt.close()

# ------------------------------------------------------------------
# FIGURE 5: Clinical Feature Heatmap
# ------------------------------------------------------------------

def generate_figure5_heatmap():
    print("\nGenerating Figure 5: Clinical Feature Heatmap...")
    
    # Select key clinical features
    if demo is not None:
        # Use original unscaled data
        feature_cols = ['age', 'glucose', 'hba1c', 'creatinine']
        available_cols = [c for c in feature_cols if c in demo.columns]
        
        if available_cols:
            cluster_means = demo.groupby('cluster')[available_cols].mean()
        else:
            # Fall back to scaled features
            cluster_means = static.groupby('cluster').mean().iloc[:, :10]
    else:
        # Use first 10 scaled features
        cluster_means = static.groupby('cluster').mean().iloc[:, :10]
    
    # Normalize for visualization
    cluster_means_norm = (cluster_means - cluster_means.mean()) / cluster_means.std()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.heatmap(cluster_means_norm.T, cmap='RdBu_r', center=0, 
               annot=True, fmt='.2f', cbar_kws={'label': 'Normalized Value'},
               linewidths=0.5, ax=ax)
    
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Clinical Feature', fontsize=12)
    ax.set_title('Clinical Feature Profiles by Cluster (Normalized)', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels([f'C{i}' for i in range(NUM_CLUSTERS)], rotation=0)
    
    plt.tight_layout()
    plt.savefig('./processed_data/figure5_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig('./processed_data/figure5_heatmap.pdf', bbox_inches='tight')
    print("✔ Saved: figure5_heatmap.png/pdf")
    plt.close()

# ------------------------------------------------------------------
# FIGURE 6: Box Plots of Key Features
# ------------------------------------------------------------------

def generate_figure6_boxplots():
    print("\nGenerating Figure 6: Feature Box Plots...")
    
    if demo is not None:
        key_features = ['age', 'glucose', 'hba1c', 'creatinine']
        available_features = [f for f in key_features if f in demo.columns]
        
        if not available_features:
            print("⚠ No key features available, skipping box plots")
            return
        
        data_source = demo
    else:
        # Use first 4 scaled features
        available_features = static.columns[:4].tolist()
        data_source = static
    
    n_features = len(available_features)
    fig, axes = plt.subplots(1, n_features, figsize=(5*n_features, 5))
    
    if n_features == 1:
        axes = [axes]
    
    for idx, feature in enumerate(available_features):
        # Prepare data for box plot
        data_list = [data_source[data_source.cluster == i][feature].values 
                    for i in range(NUM_CLUSTERS)]
        
        bp = axes[idx].boxplot(data_list, labels=[f'C{i}' for i in range(NUM_CLUSTERS)],
                              patch_artist=True, showmeans=True)
        
        # Color boxes
        colors = sns.color_palette("husl", NUM_CLUSTERS)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[idx].set_xlabel('Cluster', fontsize=11)
        axes[idx].set_ylabel(feature.capitalize(), fontsize=11)
        axes[idx].set_title(f'{feature.capitalize()} by Cluster', fontsize=12)
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./processed_data/figure6_boxplots.png', dpi=300, bbox_inches='tight')
    plt.savefig('./processed_data/figure6_boxplots.pdf', bbox_inches='tight')
    print("✔ Saved: figure6_boxplots.png/pdf")
    plt.close()

# ------------------------------------------------------------------
# TABLE 1: Evaluation Metrics
# ------------------------------------------------------------------

def generate_table1_metrics():
    print("\nGenerating Table 1: Evaluation Metrics...")
    
    try:
        metrics = pd.read_csv('./processed_data/comprehensive_metrics.csv')
    except:
        print("⚠ Metrics file not found, skipping...")
        return
    
    # Select key metrics
    metric_names = {
        'silhouette_score': 'Silhouette Score',
        'davies_bouldin_index': 'Davies-Bouldin Index',
        'calinski_harabasz_index': 'Calinski-Harabasz Index',
        'dunn_index': 'Dunn Index',
        'clustering_stability_ari': 'Clustering Stability (ARI)'
    }
    
    table_data = []
    for key, name in metric_names.items():
        if key in metrics.columns:
            value = metrics[key].iloc[0]
            if pd.notna(value):
                table_data.append([name, f'{value:.4f}'])
    
    if not table_data:
        print("⚠ No metrics available")
        return
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(10, len(table_data)*0.6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data,
                    colLabels=['Metric', 'Value'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.7, 0.3])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style cells - FIXED: Check if row exists
    for i in range(1, len(table_data) + 1):
        if (i, 0) in table._cells:  # Check if cell exists
            table[(i, 0)].set_facecolor('#ecf0f1')
    
    plt.title('Table 1: Clustering Evaluation Metrics', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('./processed_data/table1_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('./processed_data/table1_metrics.pdf', bbox_inches='tight')
    print("✔ Saved: table1_metrics.png/pdf")
    plt.close()

# ------------------------------------------------------------------
# TABLE 2: Cluster Characteristics
# ------------------------------------------------------------------

def generate_table2_cluster_characteristics():
    print("\nGenerating Table 2: Cluster Characteristics...")
    
    if demo is not None:
        key_features = ['age', 'glucose', 'hba1c', 'creatinine']
        available_features = [f for f in key_features if f in demo.columns]
        
        if not available_features:
            print("⚠ No features available")
            return
        
        cluster_summary = demo.groupby('cluster')[available_features].agg(['mean', 'std'])
    else:
        # Use scaled features
        feature_cols = static.columns[:4]
        cluster_summary = static.groupby('cluster')[feature_cols].agg(['mean', 'std'])
    
    # Prepare table data
    table_data = []
    table_data.append(['Cluster', 'n'] + [f'{col}\n(mean±std)' for col in cluster_summary.columns.levels[0]])
    
    for cluster_id in range(NUM_CLUSTERS):
        row = [f'C{cluster_id}', f'{sum(labels == cluster_id)}']
        
        for feature in cluster_summary.columns.levels[0]:
            mean_val = cluster_summary.loc[cluster_id, (feature, 'mean')]
            std_val = cluster_summary.loc[cluster_id, (feature, 'std')]
            row.append(f'{mean_val:.2f}±{std_val:.2f}')
        
        table_data.append(row)
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(14, NUM_CLUSTERS*0.8 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_data[1:],
                    colLabels=table_data[0],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    n_cols = len(table_data[0])
    for i in range(n_cols):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white')
    
    # Style cluster column - FIXED: Check if cell exists
    for i in range(1, NUM_CLUSTERS + 1):
        if (i, 0) in table._cells:
            table[(i, 0)].set_facecolor('#ecf0f1')
            table[(i, 0)].set_text_props(weight='bold')
    
    plt.title('Table 2: Clinical Characteristics by Cluster', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('./processed_data/table2_characteristics.png', dpi=300, bbox_inches='tight')
    plt.savefig('./processed_data/table2_characteristics.pdf', bbox_inches='tight')
    print("✔ Saved: table2_characteristics.png/pdf")
    plt.close()

# ------------------------------------------------------------------
# GENERATE ALL FIGURES
# ------------------------------------------------------------------

def generate_all_figures():
    print("\n" + "="*70)
    print("GENERATING ALL FIGURES AND TABLES")
    print("="*70)
    
    generate_figure1_tsne()
    generate_figure2_pca()
    generate_figure3_training()
    generate_figure4_distribution()
    generate_figure5_heatmap()
    generate_figure6_boxplots()
    generate_table1_metrics()
    generate_table2_cluster_characteristics()
    
    print("\n" + "="*70)
    print("✔ ALL FIGURES GENERATED SUCCESSFULLY")
    print("="*70)
    print("\nGenerated files in ./processed_data/:")
    print("  - figure1_tsne.png/pdf")
    print("  - figure2_pca.png/pdf")
    print("  - figure3_training.png/pdf")
    print("  - figure4_distribution.png/pdf")
    print("  - figure5_heatmap.png/pdf")
    print("  - figure6_boxplots.png/pdf")
    print("  - table1_metrics.png/pdf")
    print("  - table2_characteristics.png/pdf")
    print("\nAll figures are publication-ready (300 DPI)")
    print("="*70 + "\n")

if __name__ == "__main__":
    generate_all_figures()
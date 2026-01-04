"""
FCDT-TPFF Master Script
Complete end-to-end pipeline for paper reproduction

Optimized for 32GB RAM system

Usage:
    python fcdt_master_script.py --mimic_path /path/to/mimic-iv-2.2/

Steps:
    1. Data extraction from MIMIC-IV
    2. Feature engineering & multi-scale decomposition
    3. Model training
    4. Evaluation & baseline comparisons
    5. Figure generation

Author: Research Team
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for the entire pipeline"""
    
    def __init__(self, mimic_path):
        # Paths
        self.MIMIC_PATH = mimic_path
        self.OUTPUT_PATH = './processed_data/'
        self.FIGURES_PATH = './figures/'
        self.MODELS_PATH = './models/'
        
        # Create directories
        os.makedirs(self.OUTPUT_PATH, exist_ok=True)
        os.makedirs(self.FIGURES_PATH, exist_ok=True)
        os.makedirs(self.MODELS_PATH, exist_ok=True)
        
        # Cohort selection
        self.MIN_GLUCOSE_MEASUREMENTS = 3
        self.MIN_FOLLOWUP_MONTHS = 6
        self.MIN_AGE = 18
        
        # Model hyperparameters
        self.TEMPORAL_INPUT_DIM = 8
        self.STATIC_INPUT_DIM = 10
        self.LSTM_HIDDEN_DIM = 128
        self.GAT_HIDDEN_DIM = 128
        self.GAT_OUTPUT_DIM = 64
        self.NUM_CLUSTERS = 4
        self.DTW_WEIGHT = 0.5
        
        # Training parameters
        self.BATCH_SIZE = 32  # Optimized for 32GB RAM
        self.NUM_EPOCHS = 50  # Reduced for memory efficiency
        self.LEARNING_RATE = 0.001
        self.PATIENCE = 10
        
        # Memory optimization
        self.CHUNK_SIZE = 10000
        self.MAX_PATIENTS_IN_MEMORY = 1000
        
        # Random seed
        self.RANDOM_SEED = 42

# ============================================================================
# PIPELINE STAGES
# ============================================================================

def stage1_data_extraction(config):
    """
    Stage 1: Extract and preprocess data from MIMIC-IV
    """
    print("\n" + "=" * 80)
    print("STAGE 1: DATA EXTRACTION")
    print("=" * 80)
    
    # Import data extraction module
    sys.path.append('.')
    from fcdt_tpff_data import main as extract_main
    
    # Update config in module
    import fcdt_tpff_data
    fcdt_tpff_data.config.MIMIC_PATH = config.MIMIC_PATH
    fcdt_tpff_data.config.OUTPUT_PATH = config.OUTPUT_PATH
    
    # Run extraction
    extract_main()
    
    print("\n✓ Stage 1 Complete!")

def stage2_feature_engineering(config):
    """
    Stage 2: Feature engineering and multi-scale decomposition
    """
    print("\n" + "=" * 80)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 80)
    
    from fcdt_tpff_features import engineer_features
    
    patient_features, static_features = engineer_features(config.OUTPUT_PATH)
    
    print("\n✓ Stage 2 Complete!")
    return patient_features, static_features

def stage3_model_training(config):
    """
    Stage 3: Train FCDT-TPFF model
    """
    print("\n" + "=" * 80)
    print("STAGE 3: MODEL TRAINING")
    print("=" * 80)
    
    from fcdt_tpff_training import main_training_and_evaluation
    
    model, results, labels, embeddings = main_training_and_evaluation()
    
    print("\n✓ Stage 3 Complete!")
    return model, results, labels, embeddings

def stage4_figure_generation(config, embeddings=None, labels=None):
    """
    Stage 4: Generate all figures for paper
    """
    print("\n" + "=" * 80)
    print("STAGE 4: FIGURE GENERATION")
    print("=" * 80)
    
    from fcdt_tpff_figures import generate_all_figures
    
    generate_all_figures()
    
    print("\n✓ Stage 4 Complete!")

# ============================================================================
# QUICK START OPTIONS
# ============================================================================

def run_from_scratch(config):
    """
    Run complete pipeline from scratch
    """
    print("\n" + "=" * 80)
    print("RUNNING COMPLETE FCDT-TPFF PIPELINE")
    print("=" * 80)
    print(f"MIMIC-IV Path: {config.MIMIC_PATH}")
    print(f"Output Path: {config.OUTPUT_PATH}")
    print(f"Batch Size: {config.BATCH_SIZE} (optimized for 32GB RAM)")
    print("=" * 80)
    
    # Stage 1: Data extraction
    stage1_data_extraction(config)
    
    # Stage 2: Feature engineering
    patient_features, static_features = stage2_feature_engineering(config)
    
    # Stage 3: Model training
    model, results, labels, embeddings = stage3_model_training(config)
    
    # Stage 4: Figure generation
    stage4_figure_generation(config, embeddings, labels)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print("\nResults summary:")
    print(f"  - Processed data: {config.OUTPUT_PATH}")
    print(f"  - Figures: {config.FIGURES_PATH}")
    print(f"  - Models: {config.MODELS_PATH}")

def run_from_checkpoint(config, checkpoint='features'):
    """
    Resume from a checkpoint
    
    Args:
        checkpoint: 'features', 'training', or 'figures'
    """
    print(f"\nResuming from checkpoint: {checkpoint}")
    
    if checkpoint == 'features':
        patient_features, static_features = stage2_feature_engineering(config)
        model, results, labels, embeddings = stage3_model_training(config)
        stage4_figure_generation(config, embeddings, labels)
        
    elif checkpoint == 'training':
        model, results, labels, embeddings = stage3_model_training(config)
        stage4_figure_generation(config, embeddings, labels)
        
    elif checkpoint == 'figures':
        stage4_figure_generation(config)
        
    else:
        raise ValueError(f"Unknown checkpoint: {checkpoint}")
    
    print("\nPipeline resumed and completed!")

# ============================================================================
# MEMORY MONITORING
# ============================================================================

def check_system_requirements():
    """
    Check if system meets minimum requirements
    """
    import psutil
    
    print("\n" + "=" * 80)
    print("SYSTEM REQUIREMENTS CHECK")
    print("=" * 80)
    
    # Check RAM
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"Total RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 30:
        print("⚠ WARNING: Less than 32GB RAM detected!")
        print("  Pipeline may run slower or fail on large cohorts.")
        print("  Consider reducing BATCH_SIZE or MAX_PATIENTS_IN_MEMORY")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    else:
        print("✓ RAM requirement met")
    
    # Check disk space
    disk = psutil.disk_usage('.')
    disk_gb_free = disk.free / (1024**3)
    print(f"Free disk space: {disk_gb_free:.1f} GB")
    
    if disk_gb_free < 50:
        print("⚠ WARNING: Less than 50GB free disk space!")
        print("  MIMIC-IV processing requires substantial storage.")
    else:
        print("✓ Disk space adequate")
    
    # Check CUDA
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠ CUDA not available - will use CPU (slower)")
    
    print("=" * 80)

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_mimic_path(mimic_path):
    """
    Validate that MIMIC-IV path exists and contains required files
    Supports BOTH .csv and .csv.gz (Windows-safe)
    """
    if not os.path.isdir(mimic_path):
        raise ValueError(f"MIMIC-IV path does not exist: {mimic_path}")

    def exists_csv_or_gz(rel_path):
        base = os.path.join(mimic_path, rel_path)
        return (
            os.path.exists(base + ".csv") or
            os.path.exists(base + ".csv.gz")
        )

    required = [
        "hosp/diagnoses_icd",
        "hosp/patients",
        "hosp/admissions",
        "hosp/labevents",
        "hosp/prescriptions",
        "icu/chartevents"
    ]

    missing = [f for f in required if not exists_csv_or_gz(f)]

    if missing:
        print("\n⚠ Missing required MIMIC-IV files:")
        for f in missing:
            print(f"  - {f}.csv / {f}.csv.gz")
        raise ValueError("Incomplete MIMIC-IV dataset")

    print("✓ MIMIC-IV path validated (csv + csv.gz supported)")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """
    Main entry point with command line arguments
    """
    parser = argparse.ArgumentParser(
        description='FCDT-TPFF: Feature-Coupled Dynamic Trajectory Phenotype Fusion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python fcdt_master_script.py --mimic_path /data/mimic-iv-2.2/
  
  # Resume from feature engineering
  python fcdt_master_script.py --mimic_path /data/mimic-iv-2.2/ --resume features
  
  # Generate figures only
  python fcdt_master_script.py --figures_only
  
  # Skip system check
  python fcdt_master_script.py --mimic_path /data/mimic-iv-2.2/ --skip_check
        """
    )
    
    parser.add_argument('--mimic_path', type=str, default=None,
                       help='Path to MIMIC-IV dataset directory')
    parser.add_argument('--output_path', type=str, default='./processed_data/',
                       help='Output directory for processed data')
    parser.add_argument('--resume', type=str, choices=['features', 'training', 'figures'],
                       help='Resume from checkpoint')
    parser.add_argument('--figures_only', action='store_true',
                       help='Generate figures only (requires existing results)')
    parser.add_argument('--skip_check', action='store_true',
                       help='Skip system requirements check')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs (default: 50)')
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 80)
    print("FCDT-TPFF: Feature-Coupled Dynamic Trajectory Phenotype Fusion")
    print("Trajectory-Aware Subtyping of Type 2 Diabetes")
    print("=" * 80)
    
    # Check system requirements
    if not args.skip_check:
        check_system_requirements()
    
    # Handle figures-only mode
    if args.figures_only:
        print("\nGenerating figures only...")
        from fcdt_tpff_figures import generate_all_figures
        generate_all_figures()
        print("\n✓ Figures generated successfully!")
        return
    
    # Validate MIMIC path
    if args.mimic_path is None:
        print("\nError: --mimic_path is required")
        print("Usage: python fcdt_master_script.py --mimic_path /path/to/mimic-iv-2.2/")
        sys.exit(1)
    
    validate_mimic_path(args.mimic_path)
    
    # Initialize configuration
    config = Config(args.mimic_path)
    config.OUTPUT_PATH = args.output_path
    config.BATCH_SIZE = args.batch_size
    config.NUM_EPOCHS = args.num_epochs
    
    # Run pipeline
    if args.resume:
        run_from_checkpoint(config, args.resume)
    else:
        run_from_scratch(config)
    
    print("\n" + "=" * 80)
    print("SUCCESS! All processing complete.")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review figures in: ./figures/")
    print("2. Check results in: ./processed_data/evaluation_results.csv")
    print("3. Insert figures into your LaTeX paper")
    print("4. Cite: [Your paper citation]")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
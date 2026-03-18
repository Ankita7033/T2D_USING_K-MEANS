"""
FCDT-TPFF – Feature Engineering (FIXED - CONSISTENT DIMENSIONS)

✔ Handles variable lab availability
✔ Consistent feature dimensions across patients
✔ No dimension mismatch errors
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# MULTI-SCALE DECOMPOSER (FIXED)
# =============================================================================

class MultiScaleDecomposer:
    def __init__(self, micro=24, meso=168, macro=720):
        self.micro = micro
        self.meso = meso
        self.macro = macro

    def _aggregate(self, df, hours, features):
        """Aggregate with padding to ensure consistent dimensions"""
        df = df.copy()
        df["bin"] = (df["hours"] // hours).astype(int)
        
        # Create aggregation dict for available features only
        available_feats = [f for f in features if f in df.columns]
        
        agg_dict = {}
        for feat in available_feats:
            agg_dict[feat] = ['mean', 'std']
        
        out = df.groupby("bin")[available_feats].agg(['mean', 'std']).fillna(0)
        
        # Flatten column names
        out.columns = ['_'.join(col).strip() for col in out.columns.values]
        
        return out.values, list(out.columns)

    def decompose(self, df, features):
        """Decompose with feature tracking"""
        df = df.sort_values("charttime").reset_index(drop=True)

        t0 = df["charttime"].iloc[0]
        df["hours"] = (df["charttime"] - t0).dt.total_seconds() / 3600.0

        micro_data, micro_feats = self._aggregate(df, self.micro, features)
        meso_data, meso_feats = self._aggregate(df, self.meso, features)
        macro_data, macro_feats = self._aggregate(df, self.macro, features)

        return {
            "micro": micro_data,
            "meso": meso_data,
            "macro": macro_data,
            "feature_names": micro_feats  # Store feature names
        }


# =============================================================================
# ENHANCED FEATURE ENGINEER (FIXED)
# =============================================================================

class EnhancedFeatureEngineer:
    # Always use ONLY core temporal features for consistency
    CORE_TEMPORAL = ["glucose", "hba1c", "creatinine"]
    
    def __init__(self, labs, demographics, bmi=None, 
                 medications=None, comorbidities=None):
        self.labs = labs.copy()
        self.demo = demographics.copy()
        self.bmi = bmi.copy() if bmi is not None else None
        self.medications = medications.copy() if medications is not None else None
        self.comorbidities = comorbidities.copy() if comorbidities is not None else None

        self.static_scaler = RobustScaler()
        self.temporal_scaler = RobustScaler()
        
        # Track expected temporal feature dimension
        self.expected_temporal_dim = len(self.CORE_TEMPORAL) * 2  # mean + std

    # -------------------------------------------------------------------------
    # STATIC FEATURES
    # -------------------------------------------------------------------------

    def create_static(self):
        """Create comprehensive static feature set"""
        df = self.demo[["subject_id", "age", "gender"]].copy()
        
        # Basic demographics
        df["sex"] = (df["gender"] == "M").astype(int)
        df.drop(columns="gender", inplace=True)
        
        # Add hospital utilization if available
        if "num_admissions" in self.demo.columns:
            df["num_admissions"] = self.demo["num_admissions"]
            df["avg_los"] = self.demo["avg_los"]

        # Baseline lab values (first measurement) - ALL LABS
        baseline = (
            self.labs
            .sort_values("charttime")
            .groupby(["subject_id", "lab"])["valuenum"]
            .first()
            .unstack()
        )
        df = df.merge(baseline, on="subject_id", how="left")

        # Lab variability (CV) - ALL LABS
        lab_cv = (
            self.labs
            .groupby(["subject_id", "lab"])["valuenum"]
            .agg(lambda x: x.std() / (x.mean() + 1e-8) if len(x) > 1 else 0)
            .unstack()
        )
        lab_cv.columns = [f"{col}_cv" for col in lab_cv.columns]
        df = df.merge(lab_cv, on="subject_id", how="left")

        # Temporal trends (CORE TEMPORAL ONLY)
        trends = self._calculate_temporal_trends()
        df = df.merge(trends, on="subject_id", how="left")

        # BMI
        if self.bmi is not None:
            df = df.merge(self.bmi, on="subject_id", how="left")

        # Medication features
        if self.medications is not None:
            df = df.merge(self.medications, on="subject_id", how="left")

        # Comorbidities
        if self.comorbidities is not None:
            df = df.merge(self.comorbidities, on="subject_id", how="left")

        df.set_index("subject_id", inplace=True)

        # Imputation and scaling
        imputer = SimpleImputer(strategy="median")
        df[:] = imputer.fit_transform(df)
        df[:] = self.static_scaler.fit_transform(df)

        return df

    def _calculate_temporal_trends(self):
        """Calculate linear trends for CORE temporal features only"""
        trends = []
        
        for pid in self.labs.subject_id.unique():
            patient_data = self.labs[self.labs.subject_id == pid]
            
            features = {"subject_id": pid}
            
            # Only use CORE_TEMPORAL to ensure consistency
            for lab in self.CORE_TEMPORAL:
                lab_data = patient_data[patient_data.lab == lab].sort_values("charttime")
                
                if len(lab_data) >= 3:
                    values = lab_data["valuenum"].values
                    x = np.arange(len(values))
                    
                    # Linear trend
                    try:
                        slope, _, _, _, _ = stats.linregress(x, values)
                        features[f"{lab}_trend"] = slope
                    except:
                        features[f"{lab}_trend"] = 0
                    
                    # Rate of change
                    features[f"{lab}_roc"] = np.mean(np.abs(np.diff(values)))
                else:
                    features[f"{lab}_trend"] = 0
                    features[f"{lab}_roc"] = 0
            
            trends.append(features)
        
        return pd.DataFrame(trends)

    # -------------------------------------------------------------------------
    # TEMPORAL FEATURES (STRICT CONSISTENCY)
    # -------------------------------------------------------------------------

    def build_temporal(self, pid):
        """Build temporal features with FIXED feature set"""
        df = self.labs[self.labs.subject_id == pid]
        if len(df) < 5:
            return None

        # ALWAYS use only CORE_TEMPORAL for consistency
        wide = (
            df.pivot_table(
                index="charttime",
                columns="lab",
                values="valuenum",
                aggfunc="mean"
            )
            .reindex(columns=self.CORE_TEMPORAL)  # Force these columns
            .dropna(how="all")
            .reset_index()
        )

        if len(wide) < 5:
            return None

        # Fill missing values
        for col in self.CORE_TEMPORAL:
            if col in wide.columns:
                wide[col] = wide[col].fillna(method="ffill").fillna(method="bfill").fillna(0)

        return wide

    # -------------------------------------------------------------------------
    # ENGINEER ALL (FIXED)
    # -------------------------------------------------------------------------

    def engineer_all(self, patient_ids, decomposer):
        static = self.create_static()
        patient_ids = [p for p in patient_ids if p in static.index]

        temporal = {}
        failed = 0

        print(f"Processing {len(patient_ids)} patients...")

        for pid in patient_ids:
            df = self.build_temporal(pid)
            if df is None:
                failed += 1
                continue

            # Always use CORE_TEMPORAL
            feats = decomposer.decompose(df, self.CORE_TEMPORAL)

            # Validation
            if (
                feats["micro"].shape[0] < 2 or
                feats["meso"].shape[0] < 2 or
                feats["macro"].shape[0] < 2
            ):
                failed += 1
                continue

            # Verify consistent dimensions
            if feats["micro"].shape[1] != self.expected_temporal_dim:
                failed += 1
                continue

            temporal[pid] = feats

        print(f"✔ Temporal patients: {len(temporal)}")
        print(f"✖ Failed patients: {failed}")

        return temporal, static

    # -------------------------------------------------------------------------
    # NORMALIZATION (FIXED)
    # -------------------------------------------------------------------------

    def normalize(self, temporal, static):
        print("Normalizing temporal features (global)...")

        # Collect all micro-scale data
        all_micro = []
        for v in temporal.values():
            all_micro.append(v["micro"])
        
        all_micro = np.vstack(all_micro)
        all_micro = np.nan_to_num(all_micro)

        # Fit scaler on micro scale
        self.temporal_scaler.fit(all_micro)

        # Transform all scales
        for pid in temporal:
            for k in ["micro", "meso", "macro"]:
                data = temporal[pid][k]
                data = np.nan_to_num(data)
                temporal[pid][k] = self.temporal_scaler.transform(data)

        print("✔ Temporal normalization complete")
        
        # Verify dimensions
        sample = next(iter(temporal.values()))
        print(f"✔ Temporal feature dimension: {sample['micro'].shape[1]} (expected: {self.expected_temporal_dim})")
        
        return temporal, static


# =============================================================================
# MAIN
# =============================================================================

def engineer_features(output_path="./processed_data"):
    """Main feature engineering pipeline"""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING PIPELINE")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    labs = pd.read_csv(f"{output_path}/labs.csv", parse_dates=["charttime"])
    demo = pd.read_csv(f"{output_path}/demographics.csv")
    cohort = pd.read_csv(f"{output_path}/final_cohort.csv")

    print(f"Loaded {len(labs):,} lab records for {len(cohort)} patients")

    # Load optional features
    try:
        bmi = pd.read_csv(f"{output_path}/bmi.csv")
        print(f"✔ Loaded BMI for {len(bmi)} patients")
    except:
        bmi = None
        print("⚠ BMI data not found")

    try:
        medications = pd.read_csv(f"{output_path}/medication_features.csv")
        print(f"✔ Loaded medication features")
    except:
        medications = None
        print("⚠ Medication features not found")

    try:
        comorbidities = pd.read_csv(f"{output_path}/comorbidities.csv")
        print(f"✔ Loaded comorbidity data")
    except:
        comorbidities = None
        print("⚠ Comorbidity data not found")

    # Initialize engineer
    print("\nInitializing feature engineer...")
    fe = EnhancedFeatureEngineer(
        labs, demo, bmi, medications, comorbidities
    )
    
    decomposer = MultiScaleDecomposer()

    # Engineer features
    print("\nEngineering features...")
    temporal, static = fe.engineer_all(
        cohort.subject_id.tolist(), decomposer
    )

    print("\nNormalizing features...")
    temporal, static = fe.normalize(temporal, static)

    # Save
    print("\nSaving features...")
    with open(f"{output_path}/patient_features.pkl", "wb") as f:
        pickle.dump(temporal, f)

    static.to_csv(f"{output_path}/static_features.csv")

    # Print summary
    print("\n" + "="*60)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*60)
    print(f"Patients with features: {len(temporal)}")
    print(f"Static feature dimensions: {static.shape}")
    
    if temporal:
        sample_temporal = next(iter(temporal.values()))
        print(f"Temporal feature dimensions:")
        print(f"  Micro: {sample_temporal['micro'].shape}")
        print(f"  Meso: {sample_temporal['meso'].shape}")
        print(f"  Macro: {sample_temporal['macro'].shape}")
    
    print(f"\nFiles saved to: {output_path}/")
    print("  - patient_features.pkl")
    print("  - static_features.csv")
    print("="*60)

    return temporal, static


if __name__ == "__main__":
    engineer_features()

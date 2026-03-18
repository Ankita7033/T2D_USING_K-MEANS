"""
FCDT-TPFF: Enhanced Data Extraction Pipeline
✔ More lab features for richer temporal patterns
✔ Clinical severity indicators
✔ Medication history features
✔ Better temporal coverage filtering
✔ Coherent with improved model architecture
"""

import os
import gc
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    MIMIC_PATH = "mimic-iv"
    OUTPUT_PATH = "./processed_data"

    # Enhanced minimum requirements
    MIN_GLUCOSE_MEASUREMENTS = 5  # Increased for better temporal modeling
    MIN_TEMPORAL_SPAN_DAYS = 7    # NEW: Ensure sufficient temporal coverage
    MIN_AGE = 18
    MAX_AGE = 90  # NEW: Exclude extreme ages

    CHUNK_SIZE = 10_000
    RANDOM_SEED = 42

config = Config()

# =============================================================================
# HELPERS
# =============================================================================

def resolve_csv(base_path: str) -> str:
    """Resolve .csv or .csv.gz automatically"""
    if os.path.exists(base_path + ".csv"):
        return base_path + ".csv"
    if os.path.exists(base_path + ".csv.gz"):
        return base_path + ".csv.gz"
    raise FileNotFoundError(f"Missing file: {base_path}.csv / .csv.gz")

def read_csv_chunked(path, chunksize):
    """Universal CSV reader"""
    return pd.read_csv(
        path,
        chunksize=chunksize,
        compression="infer",
        low_memory=False
    )

# =============================================================================
# STEP 1: T2DM COHORT (UNCHANGED - ALREADY GOOD)
# =============================================================================

def extract_t2dm_cohort(mimic_path):
    print("\nSTEP 1: EXTRACTING T2DM COHORT")

    diagnoses_path = resolve_csv(
        os.path.join(mimic_path, "hosp", "diagnoses_icd")
    )

    t1_exclude_9 = {
        '250.01','250.03','250.11','250.13','250.21','250.23','250.31','250.33',
        '250.41','250.43','250.51','250.53','250.61','250.63','250.71','250.73',
        '250.81','250.83','250.91','250.93'
    }

    t2dm, t1dm = set(), set()

    for chunk in read_csv_chunked(diagnoses_path, config.CHUNK_SIZE):
        chunk["icd_code"] = chunk["icd_code"].astype(str).fillna("")

        t2_mask = (
            chunk["icd_code"].str.startswith("E11") |
            (chunk["icd_code"].str.startswith("250") &
             ~chunk["icd_code"].isin(t1_exclude_9))
        )
        t1_mask = (
            chunk["icd_code"].str.startswith("E10") |
            chunk["icd_code"].isin(t1_exclude_9)
        )

        t2dm.update(chunk.loc[t2_mask, "subject_id"])
        t1dm.update(chunk.loc[t1_mask, "subject_id"])

    cohort = list(t2dm - t1dm)
    print(f"✔ T2DM patients identified: {len(cohort)}")
    return cohort

# =============================================================================
# STEP 2: ENHANCED DEMOGRAPHICS
# =============================================================================

def extract_patient_demographics(mimic_path, patient_ids):
    print("\nSTEP 2: EXTRACTING ENHANCED DEMOGRAPHICS")

    patients_path = resolve_csv(os.path.join(mimic_path, "hosp", "patients"))
    admissions_path = resolve_csv(os.path.join(mimic_path, "hosp", "admissions"))

    patients = pd.read_csv(patients_path, compression="infer")
    admissions = pd.read_csv(admissions_path, compression="infer")

    patients = patients[patients.subject_id.isin(patient_ids)]
    admissions = admissions[admissions.subject_id.isin(patient_ids)]

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

    # Calculate length of stay
    admissions["los_days"] = (
        admissions["dischtime"] - admissions["admittime"]
    ).dt.total_seconds() / 86400

    first_adm = admissions.groupby("subject_id")["admittime"].min().reset_index()
    
    # Add hospital utilization metrics
    adm_stats = admissions.groupby("subject_id").agg({
        "hadm_id": "count",  # Number of admissions
        "los_days": "mean",  # Average length of stay
    }).rename(columns={"hadm_id": "num_admissions", "los_days": "avg_los"})

    demo = patients.merge(first_adm, on="subject_id")
    demo = demo.merge(adm_stats, on="subject_id", how="left")

    # Age handling with bounds
    demo["age"] = demo["anchor_age"]
    demo = demo[(demo["age"] >= config.MIN_AGE) & (demo["age"] <= config.MAX_AGE)]

    print(f"✔ Demographics retained: {len(demo)}")
    return demo

# =============================================================================
# STEP 3: ENHANCED LAB VALUES
# =============================================================================

def extract_lab_values(mimic_path, patient_ids):
    print("\nSTEP 3: EXTRACTING ENHANCED LAB VALUES")

    # ✨ EXPANDED LAB PANEL for better phenotyping
    lab_items = {
        # Core metabolic
        "glucose": [50809, 50931],
        "hba1c": [50852],
        "creatinine": [50912],
        
        # NEW: Kidney function
        "bun": [51006],  # Blood Urea Nitrogen
        "egfr": [50920],  # Estimated GFR
        
        # NEW: Lipid panel
        "cholesterol": [50907],
        "triglycerides": [51000],
        "hdl": [50910],
        "ldl": [50911],
        
        # NEW: Liver function
        "alt": [50861],  # Alanine Aminotransferase
        "ast": [50878],  # Aspartate Aminotransferase
        
        # NEW: Electrolytes
        "sodium": [50983],
        "potassium": [50971],
    }

    labevents_path = resolve_csv(os.path.join(mimic_path, "hosp", "labevents"))
    labs = []

    for chunk in read_csv_chunked(labevents_path, 5_000):
        chunk = chunk[chunk.subject_id.isin(patient_ids)]
        chunk = chunk[chunk.itemid.isin(sum(lab_items.values(), []))]

        if chunk.empty:
            continue

        for lab, ids in lab_items.items():
            chunk.loc[chunk.itemid.isin(ids), "lab"] = lab

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        labs.append(chunk[["subject_id", "charttime", "lab", "valuenum"]])

    labs = pd.concat(labs, ignore_index=True)
    labs = labs[labs.valuenum.notna()]

    # ✨ OUTLIER FILTERING (important for stability)
    labs = filter_lab_outliers(labs)

    print(f"✔ Lab records extracted: {len(labs):,}")
    return labs

def filter_lab_outliers(labs):
    """Remove physiologically impossible values"""
    ranges = {
        "glucose": (20, 800),
        "hba1c": (3, 20),
        "creatinine": (0.1, 15),
        "bun": (1, 150),
        "egfr": (5, 150),
        "cholesterol": (50, 500),
        "triglycerides": (20, 1000),
        "hdl": (10, 150),
        "ldl": (10, 300),
        "alt": (5, 1000),
        "ast": (5, 1000),
        "sodium": (110, 160),
        "potassium": (2, 8),
    }
    
    filtered = []
    for lab, (low, high) in ranges.items():
        lab_data = labs[labs.lab == lab]
        valid = lab_data[(lab_data.valuenum >= low) & (lab_data.valuenum <= high)]
        filtered.append(valid)
    
    return pd.concat(filtered, ignore_index=True)

# =============================================================================
# STEP 4: VITAL SIGNS (UNCHANGED - ALREADY GOOD)
# =============================================================================

def extract_vital_signs(mimic_path, patient_ids):
    print("\nSTEP 4: EXTRACTING VITAL SIGNS")
    print("Reading chartevents.csv(.gz)...")

    vital_items = {
        "sbp": [220050, 220179],
        "dbp": [220051, 220180],
        "heart_rate": [220045],
        "weight": [224639, 226512],
        "height": [226730],
    }

    chartevents_path = resolve_csv(os.path.join(mimic_path, "icu", "chartevents"))
    vitals = []
    all_itemids = sum(vital_items.values(), [])

    for i, chunk in enumerate(read_csv_chunked(chartevents_path, config.CHUNK_SIZE)):
        if i % 10 == 0:
            print(f"[Vitals] processed {(i+1)*config.CHUNK_SIZE:,} rows")

        chunk = chunk[chunk.itemid.isin(all_itemids)]
        chunk = chunk[chunk.subject_id.isin(patient_ids)]

        if chunk.empty:
            continue

        for name, ids in vital_items.items():
            chunk.loc[chunk.itemid.isin(ids), "vital"] = name

        chunk["charttime"] = pd.to_datetime(chunk["charttime"], errors="coerce")
        vitals.append(chunk[["subject_id", "charttime", "vital", "valuenum"]])

        gc.collect()

    if not vitals:
        return pd.DataFrame(columns=["subject_id", "charttime", "vital", "valuenum"])

    vitals = pd.concat(vitals, ignore_index=True)
    vitals = vitals[vitals.valuenum.notna()]

    print(f"✔ Vital records extracted: {len(vitals):,}")
    return vitals

# =============================================================================
# STEP 5: BMI (UNCHANGED - ALREADY GOOD)
# =============================================================================

def calculate_bmi(vitals):
    print("\nSTEP 5: CALCULATING BMI")

    h = vitals[vitals.vital == "height"].groupby("subject_id")["valuenum"].last()
    w = vitals[vitals.vital == "weight"].groupby("subject_id")["valuenum"].last()

    bmi = pd.concat([h, w], axis=1).dropna()
    bmi.columns = ["height_cm", "weight_kg"]
    bmi["bmi"] = bmi.weight_kg / (bmi.height_cm / 100) ** 2

    bmi = bmi[(bmi.bmi >= 15) & (bmi.bmi <= 60)]

    print(f"✔ BMI calculated for {len(bmi)} patients")
    return bmi.reset_index()[["subject_id", "bmi"]]

# =============================================================================
# STEP 6: ENHANCED MEDICATIONS
# =============================================================================

def extract_medications(mimic_path, patient_ids):
    print("\nSTEP 6: EXTRACTING ENHANCED MEDICATIONS")

    prescriptions_path = resolve_csv(os.path.join(mimic_path, "hosp", "prescriptions"))
    meds = []

    # ✨ EXPANDED medication classes
    med_patterns = {
        "insulin": "insulin",
        "metformin": "metformin",
        "sulfonylurea": "glyburide|glipizide|glimepiride",
        "dpp4": "sitagliptin|saxagliptin|linagliptin",
        "sglt2": "canagliflozin|dapagliflozin|empagliflozin",
        "glp1": "exenatide|liraglutide|dulaglutide",
        "statin": "atorvastatin|simvastatin|rosuvastatin|pravastatin",
        "ace_arb": "lisinopril|enalapril|losartan|valsartan",
    }

    for chunk in read_csv_chunked(prescriptions_path, 10_000):
        chunk = chunk[chunk.subject_id.isin(patient_ids)]
        chunk["drug"] = chunk["drug"].str.lower()
        chunk["starttime"] = pd.to_datetime(chunk["starttime"], errors="coerce")

        # Match against all patterns
        pattern = "|".join(med_patterns.values())
        chunk = chunk[chunk.drug.str.contains(pattern, na=False, regex=True)]
        
        # Classify medication
        for med_class, pattern in med_patterns.items():
            mask = chunk.drug.str.contains(pattern, na=False, regex=True)
            chunk.loc[mask, "med_class"] = med_class
        
        meds.append(chunk[["subject_id", "starttime", "drug", "med_class"]])

    meds = pd.concat(meds, ignore_index=True)
    print(f"✔ Medication records extracted: {len(meds):,}")
    return meds

def create_medication_features(meds, patient_ids):
    """Create binary medication exposure features"""
    print("\nCreating medication features...")
    
    med_features = pd.DataFrame({"subject_id": patient_ids})
    
    for med_class in meds["med_class"].unique():
        exposed = meds[meds.med_class == med_class]["subject_id"].unique()
        med_features[f"med_{med_class}"] = med_features["subject_id"].isin(exposed).astype(int)
    
    print(f"✔ Medication features created: {med_features.shape[1]-1} classes")
    return med_features

# =============================================================================
# STEP 7: COMORBIDITY EXTRACTION (NEW)
# =============================================================================

def extract_comorbidities(mimic_path, patient_ids):
    print("\nSTEP 7: EXTRACTING COMORBIDITIES")
    
    diagnoses_path = resolve_csv(os.path.join(mimic_path, "hosp", "diagnoses_icd"))
    
    # Key comorbidities relevant to diabetes phenotyping
    comorbidity_codes = {
        "hypertension": ["I10", "I11", "I12", "I13", "401", "402", "403", "404", "405"],
        "ckd": ["N18", "585"],
        "cad": ["I20", "I21", "I22", "I23", "I24", "I25", "410", "411", "412", "413", "414"],
        "chf": ["I50", "428"],
        "stroke": ["I63", "I64", "434", "436"],
        "neuropathy": ["E11.4", "250.6", "357.2"],
        "retinopathy": ["E11.3", "250.5", "362.0"],
        "nephropathy": ["E11.2", "250.4", "583"],
        "obesity": ["E66", "278.0"],
    }
    
    comorbid_dict = {pid: {k: 0 for k in comorbidity_codes.keys()} 
                     for pid in patient_ids}
    
    for chunk in read_csv_chunked(diagnoses_path, config.CHUNK_SIZE):
        chunk = chunk[chunk.subject_id.isin(patient_ids)]
        chunk["icd_code"] = chunk["icd_code"].astype(str).fillna("")
        
        for comorbid, codes in comorbidity_codes.items():
            for code in codes:
                mask = chunk["icd_code"].str.startswith(code)
                for pid in chunk.loc[mask, "subject_id"]:
                    comorbid_dict[pid][comorbid] = 1
    
    comorbid_df = pd.DataFrame.from_dict(comorbid_dict, orient="index")
    comorbid_df.reset_index(inplace=True)
    comorbid_df.rename(columns={"index": "subject_id"}, inplace=True)
    
    print(f"✔ Comorbidities extracted for {len(comorbid_df)} patients")
    return comorbid_df

# =============================================================================
# STEP 8: TEMPORAL COVERAGE VALIDATION (NEW)
# =============================================================================

def validate_temporal_coverage(labs, min_span_days=7):
    """Ensure patients have sufficient temporal span for trajectory modeling"""
    print(f"\nSTEP 8: VALIDATING TEMPORAL COVERAGE (min {min_span_days} days)")
    
    coverage = labs.groupby("subject_id")["charttime"].agg(["min", "max"])
    coverage["span_days"] = (coverage["max"] - coverage["min"]).dt.total_seconds() / 86400
    
    valid_patients = coverage[coverage.span_days >= min_span_days].index
    
    print(f"✔ Patients with sufficient temporal span: {len(valid_patients)}")
    return valid_patients.tolist()

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "="*60)
    print("FCDT-TPFF ENHANCED DATA EXTRACTION PIPELINE")
    print("="*60)

    mimic_path = os.path.abspath(config.MIMIC_PATH)
    os.makedirs(config.OUTPUT_PATH, exist_ok=True)

    # Extract base cohort
    cohort = extract_t2dm_cohort(mimic_path)
    demo = extract_patient_demographics(mimic_path, cohort)
    labs = extract_lab_values(mimic_path, cohort)

    # ✨ Enhanced filtering
    # 1. Minimum glucose measurements
    glucose_counts = labs[labs.lab == "glucose"].groupby("subject_id").size()
    valid_ids = glucose_counts[glucose_counts >= config.MIN_GLUCOSE_MEASUREMENTS].index
    print(f"\n✔ After glucose filter: {len(valid_ids)} patients")

    # 2. Temporal coverage
    labs_filtered = labs[labs.subject_id.isin(valid_ids)]
    valid_ids = validate_temporal_coverage(labs_filtered, config.MIN_TEMPORAL_SPAN_DAYS)
    print(f"✔ After temporal coverage filter: {len(valid_ids)} patients")

    # Filter all data to valid cohort
    demo = demo[demo.subject_id.isin(valid_ids)]
    labs = labs[labs.subject_id.isin(valid_ids)]

    # Create final cohort
    final_cohort = demo[["subject_id"]].drop_duplicates()
    
    # Extract remaining data
    vitals = extract_vital_signs(mimic_path, final_cohort.subject_id.tolist())
    bmi = calculate_bmi(vitals)
    meds = extract_medications(mimic_path, final_cohort.subject_id.tolist())
    
    # ✨ NEW: Medication features
    med_features = create_medication_features(meds, final_cohort.subject_id.tolist())
    
    # ✨ NEW: Comorbidities
    comorbidities = extract_comorbidities(mimic_path, final_cohort.subject_id.tolist())

    # Save all data
    final_cohort.to_csv(f"{config.OUTPUT_PATH}/final_cohort.csv", index=False)
    labs.to_csv(f"{config.OUTPUT_PATH}/labs.csv", index=False)
    demo.to_csv(f"{config.OUTPUT_PATH}/demographics.csv", index=False)
    bmi.to_csv(f"{config.OUTPUT_PATH}/bmi.csv", index=False)
    meds.to_csv(f"{config.OUTPUT_PATH}/medications.csv", index=False)
    med_features.to_csv(f"{config.OUTPUT_PATH}/medication_features.csv", index=False)
    comorbidities.to_csv(f"{config.OUTPUT_PATH}/comorbidities.csv", index=False)

    # Print summary statistics
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    print(f"Final cohort size: {len(final_cohort)}")
    print(f"Total lab measurements: {len(labs):,}")
    print(f"Unique lab types: {labs['lab'].nunique()}")
    print(f"Patients with BMI: {len(bmi)}")
    print(f"Medication records: {len(meds):,}")
    print(f"Comorbidity features: {comorbidities.iloc[:, 1:].sum().sum()}")
    print(f"\nData saved to: {config.OUTPUT_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()


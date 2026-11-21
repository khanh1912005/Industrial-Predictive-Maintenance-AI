
from pathlib import Path
import subprocess
import sys


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==========================================
# 2. DEFINE PIPELINE STEPS
# ==========================================

PIPELINE_STEPS = [
    (
        "STEP 1: DATA VALIDATION",
        "src/data/validate_data.py"
    ),
    (
        "STEP 2: DATA CLEANING",
        "src/data/clean_data.py"
    ),
    (
        "STEP 3: FEATURE ANALYSIS",
        "src/features/select_features.py"
    ),
    (
        "STEP 4: RUL ENGINEERING",
        "src/features/build_rul_labels.py"
    ),
    (
        "STEP 5: DATASET SPLITTING",
        "src/data/split_data.py"
    ),
    (
        "STEP 6: FEATURE SCALING",
        "src/features/scale_features.py"
    ),
    (
        "STEP 7: TIME-SERIES WINDOWING",
        "src/features/create_windows.py"
    )
]


# ==========================================
# 3. RUN ONE PIPELINE STEP
# ==========================================

def run_step(step_name, script_path):

    print("\n" + "=" * 60)
    print(step_name)
    print("=" * 60)

    full_script_path = PROJECT_ROOT / script_path

    if not full_script_path.is_file():
        raise FileNotFoundError(
            f"Khong tim thay script: {full_script_path}"
        )

    subprocess.run(
        [
            sys.executable,
            str(full_script_path)
        ],
        cwd=PROJECT_ROOT,
        check=True
    )

    print(f"\n{step_name}: SUCCESS")


# ==========================================
# 4. VERIFY PIPELINE OUTPUTS
# ==========================================

def verify_outputs():

    required_outputs = [
        "data/processed/train_FD001_clean.csv",
        "data/processed/train_FD001_rul_capped.csv",
        "data/processed/train_FD001_model_train.csv",
        "data/processed/train_FD001_validation.csv",

        "data/processed/train_FD001_standard_scaled.csv",
        "data/processed/validation_FD001_standard_scaled.csv",

        "data/processed/train_FD001_minmax_scaled.csv",
        "data/processed/validation_FD001_minmax_scaled.csv",

        "data/processed/windows/X_train.npy",
        "data/processed/windows/y_train.npy",
        "data/processed/windows/X_validation.npy",
        "data/processed/windows/y_validation.npy",

        "artifacts/scalers/standard_scaler.joblib",
        "artifacts/scalers/minmax_scaler.joblib",
        "artifacts/scalers/feature_metadata.json",

        "reports/feature_selection_report.csv",
        "reports/high_correlation_pairs.csv"
    ]

    missing_outputs = []

    for relative_path in required_outputs:

        full_path = PROJECT_ROOT / relative_path

        if not full_path.is_file():
            missing_outputs.append(relative_path)

    print("\n===== OUTPUT VERIFICATION =====")

    if missing_outputs:

        print("MISSING OUTPUT FILES:")

        for path in missing_outputs:
            print(f"- {path}")

        raise FileNotFoundError(
            "Pipeline output verification failed."
        )

    print(f"Verified files: {len(required_outputs)}")
    print("All required outputs exist.")


# ==========================================
# 5. RUN COMPLETE PIPELINE
# ==========================================

def main():

    print("\n")
    print("*" * 60)
    print("INDUSTRIAL PREDICTIVE MAINTENANCE")
    print("PREPROCESSING PIPELINE")
    print("*" * 60)

    print(f"\nProject root: {PROJECT_ROOT}")

    try:

        for step_name, script_path in PIPELINE_STEPS:

            run_step(
                step_name,
                script_path
            )

        verify_outputs()

    except subprocess.CalledProcessError as error:

        print("\n" + "!" * 60)
        print("PIPELINE FAILED")
        print(f"Script exit code: {error.returncode}")
        print("Fix the error before continuing.")
        print("!" * 60)

        raise SystemExit(error.returncode)

    except Exception as error:

        print("\n" + "!" * 60)
        print("PIPELINE FAILED")
        print(f"Reason: {error}")
        print("!" * 60)

        raise SystemExit(1)

    print("\n" + "*" * 60)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("*" * 60)


# ==========================================
# 6. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()

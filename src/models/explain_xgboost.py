
from pathlib import Path

import joblib
import json
import pandas as pd
import shap
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "artifacts" / "models" / "xgboost.joblib"

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "validation_FD001_standard_scaled.csv"
)

METADATA_PATH = (
    BASE_DIR
    / "artifacts"
    / "scalers"
    / "feature_metadata.json"
)


def main():

    # 1. Load trained model
    model = joblib.load(MODEL_PATH)

    # 2. Load validation data
    validation_df = pd.read_csv(DATA_PATH)

    # 3. Load feature metadata
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["selected_features"]

    X_validation = validation_df[feature_columns]

    # 4. Create SHAP explainer
    explainer = shap.TreeExplainer(model)

    # 5. Calculate SHAP values
    shap_values = explainer(X_validation)

    print("===== SHAP INFORMATION =====")
    print("SHAP values shape:", shap_values.values.shape)
    print("Base value:", explainer.expected_value)

    print("\nSHAP calculation completed!")
    
    # 6. Explain one validation sample

    sample_index = 0

    print("\n===== SINGLE SAMPLE EXPLANATION =====")

    print("Actual RUL:", validation_df.iloc[sample_index]["RUL"])

    print(
        "Predicted RUL:",
        model.predict(X_validation.iloc[[sample_index]])[0]
    )

    print("\nFeature contributions:")

    sample_shap = pd.DataFrame({
        "feature": feature_columns,
        "feature_value": X_validation.iloc[sample_index].values,
        "shap_value": shap_values.values[sample_index],
    })

    sample_shap["absolute_shap"] = sample_shap["shap_value"].abs()

    sample_shap = sample_shap.sort_values(
        by="absolute_shap",
        ascending=False,
    )

    print(sample_shap.to_string(index=False))
        # 7. Global SHAP Explanation

    print("\n===== GLOBAL SHAP EXPLANATION =====")

    # Calculate mean absolute SHAP value for each feature
    mean_abs_shap = abs(shap_values.values).mean(axis=0)

    global_shap = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs_shap,
    })

    # Sort features by global importance
    global_shap = global_shap.sort_values(
        by="mean_abs_shap",
        ascending=False,
    )

    print(global_shap.to_string(index=False))

    # Save global SHAP report
    report_dir = BASE_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    global_shap.to_csv(
        report_dir / "xgboost_shap_global_importance.csv",
        index=False,
    )

    # Create global SHAP bar plot
    shap.plots.bar(
        shap_values,
        max_display=17,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        report_dir / "xgboost_shap_global_bar.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    
    # 8. Global SHAP Beeswarm Plot

    shap.plots.beeswarm(
        shap_values,
        max_display=17,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        report_dir / "xgboost_shap_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()

    plt.close()

    print("\nGlobal SHAP report and plot saved!")

if __name__ == "__main__":
    main()

import os
import random
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
from dotenv import load_dotenv
import argilla as rg

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
for lib in ["argilla", "httpx", "httpcore"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# Load environment variables
load_dotenv()

# Fix random seed for reproducible multi-annotator resolution
SEED = int(os.getenv("EVAL_SCORES_SEED", 42))
rng = random.Random(SEED)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

# Argilla Dataset Names
ARGILLA_DATASET_HUM = os.getenv("ARGILLA_DATASET_HUM", "pass-or-fail-hum")
ARGILLA_DATASET_PROT_EX = os.getenv("ARGILLA_DATASET_PROT_EX", "pass-or-fail-prot_ex")

# File Paths (Configurable via .env, fallback to repository relative paths)
HUM_CSV_PATH = Path(os.getenv("EVAL_HUM_CSV_PATH", BASE_DIR / "human_evaluation_panellinies_hum_subset.csv")).resolve()
SCI_CSV_PATH = Path(os.getenv("EVAL_SCI_CSV_PATH", BASE_DIR / "human_evaluation_panellinies_science_subset.csv")).resolve()
PROT_CSV_PATH = Path(os.getenv("EVAL_PROT_CSV_PATH", BASE_DIR / "human_evaluation_protipa_subset.csv")).resolve()

EXCEL_OUT_PATH = Path(os.getenv("EVAL_EXCEL_OUT_PATH", BASE_DIR / "evaluation_scores_master.xlsx")).resolve()
CSV_OUT_PATH = Path(os.getenv("EVAL_CSV_OUT_PATH", BASE_DIR / "evaluation_scores_master.csv")).resolve()

def extract_argilla_dataset(client, dataset_name, workspace_name):
    """Extract annotations from an Argilla dataset and resolve multiple responses with seed."""
    logging.info(f"Fetching dataset '{dataset_name}' from workspace '{workspace_name}'...")
    dataset = client.datasets(name=dataset_name, workspace=workspace_name)
    if not dataset:
        raise ValueError(f"Dataset '{dataset_name}' not found in workspace '{workspace_name}'")

    records_map = {}
    multi_response_count = 0

    for rec in dataset.records:
        meta = rec.metadata or {}
        q_id = str(meta.get("question_id", "")).strip()
        model_name = str(meta.get("model_name", "")).strip()
        subject = str(meta.get("subject", "")).strip()

        resp_dict = rec.responses.to_dict() if hasattr(rec.responses, "to_dict") else {}
        grades = resp_dict.get("grade", [])
        explanations = resp_dict.get("explanation", [])

        human_grade = None
        human_explanation = None
        user_id = None

        if grades:
            # Pair grade with matching explanation by user_id if available
            exp_by_user = {e.get("user_id"): e.get("value") for e in explanations if isinstance(e, dict)}

            if len(grades) > 1:
                multi_response_count += 1
                # Sort deterministically by user_id before random choice
                sorted_grades = sorted(grades, key=lambda x: str(x.get("user_id", "")))
                chosen = rng.choice(sorted_grades)
            else:
                chosen = grades[0]

            try:
                human_grade = float(chosen.get("value"))
            except (ValueError, TypeError):
                human_grade = None

            user_id = chosen.get("user_id")
            human_explanation = exp_by_user.get(user_id) or (explanations[0].get("value") if explanations else None)

        records_map[(q_id, model_name)] = {
            "Human_Grade": human_grade,
            "Explanation": human_explanation,
            "Annotator_ID": user_id,
            "Subject_Argilla": subject,
        }

    logging.info(f"'{dataset_name}': Extracted {len(records_map)} records ({multi_response_count} had multiple responses resolved with seed={SEED}).")
    return records_map

def compute_correlation_stats(df, group_col=None):
    """Compute Spearman, Pearson, and MAE correlation metrics on evaluated records."""
    records = []
    
    def calc_metrics(sub_df, name):
        valid = sub_df.dropna(subset=["Human_Grade", "LLM_Judge_Score"]).copy()
        n = len(valid)
        if n < 3:
            return {
                "Subset": name,
                "Sample_Count": n,
                "Spearman_r": np.nan,
                "Spearman_p": np.nan,
                "Pearson_r": np.nan,
                "Pearson_p": np.nan,
                "MAE": np.nan,
                "Mean_Human_Grade": valid["Human_Grade"].mean() if n > 0 else np.nan,
                "Mean_LLM_Score": valid["LLM_Judge_Score"].mean() if n > 0 else np.nan,
            }
        
        h = valid["Human_Grade"].astype(float).values
        m = valid["LLM_Judge_Score"].astype(float).values

        if np.std(h) == 0 or np.std(m) == 0:
            spearman_r, spearman_p = np.nan, np.nan
            pearson_r, pearson_p = np.nan, np.nan
        else:
            res_s = stats.spearmanr(h, m)
            spearman_r, spearman_p = res_s.statistic, res_s.pvalue
            res_p = stats.pearsonr(h, m)
            pearson_r, pearson_p = res_p.statistic, res_p.pvalue

        mae = float(np.mean(np.abs(h - m)))

        return {
            "Subset": name,
            "Sample_Count": n,
            "Spearman_r": round(float(spearman_r), 4) if pd.notna(spearman_r) else np.nan,
            "Spearman_p": round(float(spearman_p), 5) if pd.notna(spearman_p) else np.nan,
            "Pearson_r": round(float(pearson_r), 4) if pd.notna(pearson_r) else np.nan,
            "Pearson_p": round(float(pearson_p), 5) if pd.notna(pearson_p) else np.nan,
            "MAE": round(mae, 4),
            "Mean_Human_Grade": round(float(np.mean(h)), 4),
            "Mean_LLM_Score": round(float(np.mean(m)), 4),
        }

    # Overall metrics
    records.append(calc_metrics(df, "ALL COMBINED"))

    # Grouped metrics
    if group_col and group_col in df.columns:
        for val, group in df.groupby(group_col):
            records.append(calc_metrics(group, f"{group_col}: {val}"))

    return pd.DataFrame(records)

def main():
    load_dotenv()
    api_url = os.getenv("ARGILLA_API_URL")
    api_key = os.getenv("ARGILLA_API_KEY")
    workspace_name = os.getenv("ARGILLA_POF_WORKSPACE", "pass-or-fail")

    if not api_url or not api_key:
        raise ValueError("ARGILLA_API_URL and ARGILLA_API_KEY must be configured in .env")

    client = rg.Argilla(api_url=api_url, api_key=api_key)

    # 1. Fetch Argilla datasets
    hum_argilla = extract_argilla_dataset(client, ARGILLA_DATASET_HUM, workspace_name)
    prot_argilla = extract_argilla_dataset(client, ARGILLA_DATASET_PROT_EX, workspace_name)

    # 2. Read base CSVs
    df_hum = pd.read_csv(HUM_CSV_PATH)
    df_sci = pd.read_csv(SCI_CSV_PATH)
    df_prot = pd.read_csv(PROT_CSV_PATH)

    # 3. Enrich df_hum
    df_hum["Dataset"] = "Panellinies Humanities"
    df_hum["Source"] = "Argilla (pass-or-fail-hum)"
    df_hum["Human_Grade"] = None
    df_hum["Explanation"] = None
    df_hum["Annotator_ID"] = None

    for idx, row in df_hum.iterrows():
        key = (str(row["Question_ID"]).strip(), str(row["Model"]).strip())
        if key in hum_argilla:
            info = hum_argilla[key]
            df_hum.at[idx, "Human_Grade"] = info["Human_Grade"]
            df_hum.at[idx, "Explanation"] = info["Explanation"]
            df_hum.at[idx, "Annotator_ID"] = info["Annotator_ID"]

    # 4. Enrich df_prot
    df_prot["Dataset"] = "Protipa Exams"
    df_prot["Source"] = "Argilla (pass-or-fail-prot_ex)"
    df_prot["Human_Grade"] = None
    df_prot["Explanation"] = None
    df_prot["Annotator_ID"] = None

    for idx, row in df_prot.iterrows():
        key = (str(row["Question_ID"]).strip(), str(row["Model"]).strip())
        if key in prot_argilla:
            info = prot_argilla[key]
            df_prot.at[idx, "Human_Grade"] = info["Human_Grade"]
            df_prot.at[idx, "Explanation"] = info["Explanation"]
            df_prot.at[idx, "Annotator_ID"] = info["Annotator_ID"]

    # 5. Enrich df_sci (from local annotations)
    df_sci["Dataset"] = "Panellinies Sciences"
    df_sci["Source"] = "Local Evaluation App"
    if "Annotator_ID" not in df_sci.columns:
        df_sci["Annotator_ID"] = "local_reviewer"

    # Canonical columns order
    common_cols = [
        "Dataset",
        "Subject",
        "Model",
        "Question_ID",
        "Input_Question",
        "Reference_Target",
        "Model_Answer",
        "LLM_Judge_Score",
        "BERTScore",
        "Human_Grade",
        "Explanation",
        "Source",
        "Annotator_ID",
    ]

    df_hum_out = df_hum[common_cols].copy()
    df_sci_out = df_sci[[c for c in common_cols if c in df_sci.columns]].copy()
    df_prot_out = df_prot[common_cols].copy()

    # Master combined dataframe
    df_all = pd.concat([df_hum_out, df_sci_out, df_prot_out], ignore_index=True)

    # 6. Compute correlation tables
    corr_by_dataset = compute_correlation_stats(df_all, group_col="Dataset")
    corr_by_subject = compute_correlation_stats(df_all, group_col="Subject")
    corr_by_model = compute_correlation_stats(df_all, group_col="Model")
    corr_summary = pd.concat([corr_by_dataset, corr_by_subject[1:], corr_by_model[1:]], ignore_index=True)

    # 7. Write to CSV
    df_all.to_csv(CSV_OUT_PATH, index=False, encoding="utf-8-sig")
    logging.info(f"Wrote master CSV: {CSV_OUT_PATH} ({len(df_all)} rows)")

    # 8. Write to Excel Workbook with multiple sheets (clip strings to 32,000 chars for Excel limit)
    def clip_for_excel(df, max_len=32000):
        df_c = df.copy()
        for col in df_c.select_dtypes(include=["object", "str"]).columns:
            df_c[col] = df_c[col].map(lambda x: str(x)[:max_len] if pd.notna(x) and len(str(x)) > max_len else x)
        return df_c

    with pd.ExcelWriter(EXCEL_OUT_PATH, engine="openpyxl") as writer:
        clip_for_excel(df_all).to_excel(writer, sheet_name="All_Scores", index=False)
        clip_for_excel(df_hum_out).to_excel(writer, sheet_name="Panellinies_Humanities", index=False)
        clip_for_excel(df_sci_out).to_excel(writer, sheet_name="Panellinies_Sciences", index=False)
        clip_for_excel(df_prot_out).to_excel(writer, sheet_name="Protipa_Exams", index=False)
        corr_summary.to_excel(writer, sheet_name="Correlations", index=False)

    logging.info(f"Wrote Excel Workbook: {EXCEL_OUT_PATH}")

    # Print summary metrics to stdout
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total Records: {len(df_all)}")
    print(f"  - Panellinies Humanities: {len(df_hum_out)} (Evaluated: {df_hum_out['Human_Grade'].notna().sum()})")
    print(f"  - Panellinies Sciences:   {len(df_sci_out)} (Evaluated: {df_sci_out['Human_Grade'].notna().sum()})")
    print(f"  - Protipa Exams:          {len(df_prot_out)} (Evaluated: {df_prot_out['Human_Grade'].notna().sum()})")
    print(f"Total Evaluated Items:     {df_all['Human_Grade'].notna().sum()} / {len(df_all)}")
    print("\n=== CORRELATION ANALYSIS (HUMAN GRADE vs LLM JUDGE) ===")
    print(corr_summary.to_string(index=False))

if __name__ == "__main__":
    main()

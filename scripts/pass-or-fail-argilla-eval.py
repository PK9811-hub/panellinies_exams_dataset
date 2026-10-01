# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Pass or Fail: Argilla Workspace & Dataset Setup (Protipa Exams)
#
# ## About this Script (Jupytext Format)
# This file is structured in **Jupytext percent format** (`# %%`), allowing it to serve as both a standard, executable Python script and an interactive Jupyter notebook.
#
# ### What this Script Does
# 1. Connects to the remote Argilla instance using project credentials.
# 2. Ensures the dedicated evaluation workspace (`pass-or-fail`) exists.
# 3. Ensures annotator accounts (`prokopis`, `pkyriazi`, `ekasoura`) exist and assigns them to the workspace.
# 4. Defines the Argilla dataset schema (`pass-or-fail-prot_ex`) tailored for human evaluation:
#    - **Fields**: `Question ID`, `Context` (reading passages), `Question`, `Images`, `Image description`, `Reference answer`, and `Model answer`.
#    - **Questions**: `Grade` discrete rating scale (`0.0`, `0.25`, `0.5`, `0.75`, `1.0`) and optional Greek `Explanation`.
#    - **Metadata**: `Subject` and `Question ID` (visible to annotators); `Model` (`model_name`) is stored as metadata (hidden from annotators); LLM judge and BERT scores are omitted for blind grading.
# 5. Fetches the human evaluation subset CSV and enriches each record with passages, images, and canonical solutions from the Hugging Face dataset (`ilsp/greek-protipa-exams`).
# 6. Uploads the enriched records to Argilla.
#
# ### Converting to an `.ipynb` Notebook
# To convert this file to a Jupyter notebook (`.ipynb`), run:
# ```bash
# uv run jupytext --to notebook notebooks/pass-or-fail-argilla-eval.py
# ```
# Alternatively, open this `.py` file directly in VS Code or Jupyter Lab, and it will be recognized as an interactive notebook with runnable cells (`# %%`).
#
# ### Why We Do Not Commit `.ipynb` Notebooks
# - **Clean Git History**: Plain Python files (`.py`) produce readable, surgical line-based diffs without JSON noise.
# - **No Metadata Churn**: Avoids tracking kernel states, output caches, execution counts, and binary cell outputs in version control.
# - **Dual Usability**: Can be executed headlessly via CLI (`uv run python notebooks/pass-or-fail-argilla-eval.py`) or cell-by-cell in an IDE.
#
# ---
#
# ## Required Environment Variables (`.env`)
# This script reads configuration from the repository root `.env` file and expects the following variables:
#
# | Variable | Description | Example / Default |
# |---|---|---|
# | `ARGILLA_API_URL` | Base URL of the remote Argilla server | `https://nlp.ilsp.gr/argilla/` |
# | `ARGILLA_API_KEY` | API Key for Argilla owner/admin operations | `nfx_...` |
# | `ARGILLA_POF_WORKSPACE` | Target Argilla workspace for the project | `pass-or-fail` |
# | `ARGILLA_DATASET_PROT_EX` | Name of the Argilla evaluation dataset | `pass-or-fail-prot_ex` |
# | `PROT_EX_DATASET` | Hugging Face repository for Protipa exams | `ilsp/greek-protipa-exams` |
# | `HF_TOKEN` | Hugging Face API token for dataset access | `hf_...` |

# %%
import os
import json
import ast
import base64
from io import BytesIO
from pathlib import Path
import pandas as pd
from dotenv import dotenv_values
from datasets import load_dataset
from huggingface_hub import login
import argilla as rg

# %% [markdown]
# ## Configuration & Flags

# %%
# Flag to delete and recreate dataset if it already exists
RECREATE_DATASET = True
# Minimum number of submissions to close a record.
MIN_SUBMITTED = 1

# Locate project root and load environment variables
notebooks_dir = Path(__file__).resolve().parent if "__file__" in locals() else Path.cwd()
project_root = notebooks_dir.parent if notebooks_dir.name == "notebooks" else notebooks_dir
env_path = project_root / ".env"

env = dotenv_values(env_path)

argilla_api_url = env.get("ARGILLA_API_URL")
argilla_api_key = env.get("ARGILLA_API_KEY")
workspace_name = env.get("ARGILLA_POF_WORKSPACE", "pass-or-fail")
dataset_name = env.get("ARGILLA_DATASET_PROT_EX", "pass-or-fail-prot_ex")
prot_ex_hf_repo = env.get("PROT_EX_DATASET", "ilsp/greek-protipa-exams")
hf_token = env.get("HF_TOKEN")

if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("Logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"HF login notice: {e}")

print(f"Argilla API URL: {argilla_api_url}")
print(f"Workspace Name: {workspace_name}")
print(f"Dataset Name: {dataset_name}")
print(f"HF Dataset Repo: {prot_ex_hf_repo}")
print(f"Recreate Dataset Flag: {RECREATE_DATASET}")

client = rg.Argilla(
    api_url=argilla_api_url,
    api_key=argilla_api_key,
)

print(f"Argilla Client connected. Argilla version: {rg.__version__}")

# %% [markdown]
# ## 1. Manage Workspace
#
# Ensure workspace `pass-or-fail` exists. If not, create it.

# %%
workspace = client.workspaces(workspace_name)
if workspace is None:
    print(f"Workspace '{workspace_name}' does not exist. Creating...")
    workspace = rg.Workspace(name=workspace_name, client=client).create()
    print(f"Workspace '{workspace.name}' (ID: {workspace.id}) created.")
else:
    print(f"Workspace '{workspace.name}' (ID: {workspace.id}) found.")

# %% [markdown]
# ## 2. Manage Annotator Users
#
# Ensure annotator accounts (`prokopis`, `pkyriazi`, `ekasoura`) exist and are linked to the workspace.

# %%
annotator_usernames = ["prokopis", "pkyriazi", "ekasoura"]

for username in annotator_usernames:
    user = client.users(username)
    if user is None:
        print(f"User '{username}' not found. Creating user with password='{username}'...")
        user = rg.User(
            username=username,
            password=username,
            role="annotator",
            client=client,
        ).create()
        print(f"Created user '{username}' (ID: {user.id}).")
    else:
        print(f"User '{username}' already exists.")

    try:
        workspace.add_user(username)
        print(f"Added user '{username}' to workspace '{workspace_name}'.")
    except Exception as e:
        print(f"Notice adding '{username}' to workspace '{workspace_name}': {e}")

# %% [markdown]
# ## 3. Define Dataset Schema (Protipa Human Evaluation)
#
# Configure fields, hidden metadata properties, questions, and guidelines mirroring the LLM-as-a-judge rubric.

# %%
guidelines = """
### Οδηγίες Αξιολόγησης (Human Evaluation Rubric)

Αξιολογείς μια υποβληθείσα απάντηση (**Model answer**) σε μια ερώτηση (**Question**), συγκρίνοντάς τη με την πρότυπη απάντηση (**Reference answer**) και λαμβάνοντας υπόψη το πλαίσιο (**Context**) και τυχόν σχεδιαγράμματα/εικόνες (**Images** / **Image description**).

#### Βαθμολογική Κλίμακα (Grade):
- **1.0**: Πλήρως ορθή και ολοκληρωμένη απάντηση / λύση.
- **0.75**: Ορθή προσέγγιση με μικρές παραλείψεις ή επουσιώδη αριθμητικά/διατυπωτικά σφάλματα.
- **0.5**: Μερικώς ορθή απάντηση (κατανόηση της μεθοδολογίας, αλλά σημαντικά λάθη στην εκτέλεση ή στο αποτέλεσμα).
- **0.25**: Ελάχιστα σωστά στοιχεία / λανθασμένη συλλογιστική πορεία.
- **0.0**: Εντελώς λανθασμένη, άσχετη ή κενή απάντηση.

#### Αιτιολόγηση (Explanation):
- Σύντομη αιτιολόγηση του βαθμού στα Ελληνικά (προαιρετική εφόσον ο βαθμός είναι προφανής, απαραίτητη σε περιπτώσεις μερικής βαθμολόγησης).
"""

settings = rg.Settings(
    fields=[
        rg.TextField(name="question_id_field", title="Question ID", use_markdown=False),
        rg.TextField(name="input", title="Context", use_markdown=True, required=False),
        rg.TextField(name="question", title="Question", use_markdown=True),
        rg.TextField(name="images", title="Images", use_markdown=True, required=False),
        rg.TextField(name="image_description", title="Image description", use_markdown=True, required=False),
        rg.TextField(name="reference_answer", title="Reference answer", use_markdown=True),
        rg.TextField(name="answer", title="Model answer", use_markdown=True),
    ],
    questions=[
        rg.LabelQuestion(
            name="grade",
            title="Grade",
            labels=["0.0", "0.25", "0.5", "0.75", "1.0"],
            required=True,
        ),
        rg.TextQuestion(
            name="explanation",
            title="Explanation",
            use_markdown=True,
            required=False,
        ),
    ],
    metadata=[
        rg.TermsMetadataProperty(name="subject", title="Subject", visible_for_annotators=True),
        rg.TermsMetadataProperty(name="question_id", title="Question ID", visible_for_annotators=True),
        rg.TermsMetadataProperty(name="model_name", title="Model", visible_for_annotators=False),
        # rg.FloatMetadataProperty(name="llm_judge_score", title="LLM Judge Score", visible_for_annotators=False),
        # rg.FloatMetadataProperty(name="bert_score", title="BERTScore", visible_for_annotators=False),
    ],
    guidelines=guidelines.strip(),
    distribution=rg.TaskDistribution(min_submitted=MIN_SUBMITTED),
)

# %% [markdown]
# ## 4. Load Data & Enrich with Hugging Face Dataset

# %%
def images_to_markdown(images_list):
    """Encodes a list of PIL Images or image URLs into HTML/Markdown img elements."""
    if not images_list:
        return ""
    md_elements = []
    for idx, img in enumerate(images_list):
        if img is None:
            continue
        try:
            if hasattr(img, "save"):
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                md_elements.append(
                    f'<img src="data:image/png;base64,{b64_str}" alt="Diagram {idx+1}" style="max-width: 100%; height: auto;" />'
                )
            elif isinstance(img, str) and img.strip():
                md_elements.append(f"![Diagram {idx+1}]({img})")
        except Exception as e:
            print(f"Notice encoding image {idx}: {e}")
    return "\n\n".join(md_elements)


def parse_dict_field(val):
    """Parses a string representing a JSON object or Python dict into a dict."""
    if val is None or pd.isna(val):
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    pass
    return {}


def unpack_field(val, preferred_key=None):
    """Extracts text content from a JSON dict or raw string field."""
    parsed = parse_dict_field(val)
    if parsed:
        if preferred_key and preferred_key in parsed:
            return str(parsed[preferred_key]).strip()
        for k in ["answer", "reference", "question", "context", "text", "content"]:
            if k in parsed:
                return str(parsed[k]).strip()
        if len(parsed) == 1:
            return str(next(iter(parsed.values()))).strip()
    return str(val).strip() if pd.notna(val) else ""


# 1. Load CSV subset
csv_url = "https://raw.githubusercontent.com/PK9811-hub/protipa_exams_dataset/main/human_evaluation_protipa_subset.csv"
print(f"Fetching human evaluation CSV from {csv_url}...")
df_subset = pd.read_csv(csv_url)
print(f"Loaded {len(df_subset)} records from CSV.")
print("CSV Columns:", list(df_subset.columns))

# 2. Load Hugging Face dataset to enrich context, images, and canonical reference
print(f"Loading Hugging Face dataset '{prot_ex_hf_repo}'...")
hf_dataset_dict = load_dataset(prot_ex_hf_repo)

# Index HF dataset across splits by 'id'
hf_index = {}
for split_name in hf_dataset_dict.keys():
    for item in hf_dataset_dict[split_name]:
        hf_index[item["id"]] = item

print(f"Indexed {len(hf_index)} records from Hugging Face dataset.")

# Verify or initialize dataset in Argilla
existing_dataset = client.datasets(name=dataset_name, workspace=workspace_name)

if existing_dataset and RECREATE_DATASET:
    print(f"Deleting existing dataset '{dataset_name}' (RECREATE_DATASET=True)...")
    existing_dataset.delete()
    existing_dataset = None

if existing_dataset:
    print(f"Dataset '{dataset_name}' already exists in workspace '{workspace_name}'. Using existing dataset.")
    dataset = existing_dataset
else:
    print(f"Creating dataset '{dataset_name}' in workspace '{workspace_name}'...")
    dataset = rg.Dataset(
        name=dataset_name,
        workspace=workspace_name,
        settings=settings,
        client=client,
    ).create()
    print(f"Dataset '{dataset.name}' (ID: {dataset.id}) created.")

# Build and log enriched records
records = []
matched_hf_count = 0

for _, row in df_subset.iterrows():
    qid = str(row["Question_ID"])
    hf_item = hf_index.get(qid, {})

    # Extract CSV fields, unpacking possible JSON/dict structures
    input_q_raw = row.get("Input_Question")
    input_q_dict = parse_dict_field(input_q_raw)
    csv_question = input_q_dict.get("question", "") if input_q_dict else str(input_q_raw or "").strip()
    csv_context = input_q_dict.get("context", "") if input_q_dict else ""
    csv_ans = unpack_field(row.get("Reference_Target"), preferred_key="reference")
    model_ans = unpack_field(row.get("Model_Answer"), preferred_key="answer")

    if hf_item:
        matched_hf_count += 1
        # Extract enriched fields from HF dataset
        question_text = str(hf_item.get("question") or csv_question)
        context_input = str(hf_item.get("input") or csv_context)
        images_rendered = images_to_markdown(hf_item.get("images", []))
        image_desc = str(hf_item.get("image_description") or hf_item.get("image_transcription") or "")

        # Reference answer: use HF answer_text and/or detailed solution from CSV
        hf_ans = str(hf_item.get("answer_text") or "").strip()
        if hf_ans and csv_ans and hf_ans != csv_ans:
            ref_ans = f"**Answer Key:** {hf_ans}\n\n**Detailed Solution:**\n{csv_ans}"
        else:
            ref_ans = hf_ans or csv_ans
    else:
        # Fallback to CSV fields if ID is not found in HF dataset
        question_text = csv_question
        context_input = csv_context
        images_rendered = ""
        image_desc = ""
        ref_ans = csv_ans

    metadata = {
        "subject": str(row["Subject"]) if pd.notna(row["Subject"]) else "",
        "model_name": str(row["Model"]) if pd.notna(row["Model"]) else "",
        "question_id": qid,
    }
    # if pd.notna(row.get("LLM_Judge_Score")):
    #     try:
    #         metadata["llm_judge_score"] = float(row["LLM_Judge_Score"])
    #     except ValueError:
    #         pass
    # if pd.notna(row.get("BERTScore")):
    #     try:
    #         metadata["bert_score"] = float(row["BERTScore"])
    #     except ValueError:
    #         pass

    record = rg.Record(
        fields={
            "question_id_field": qid,
            "input": context_input,
            "question": question_text,
            "images": images_rendered,
            "image_description": image_desc,
            "reference_answer": ref_ans,
            "answer": model_ans,
        },
        metadata=metadata,
    )
    records.append(record)

print(f"Matched {matched_hf_count} / {len(df_subset)} CSV records with HF dataset.")
print(f"Logging {len(records)} records to Argilla dataset '{dataset_name}'...")
dataset.records.log(records)
print(f"Successfully logged {len(records)} records.")
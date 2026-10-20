# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Pass or Fail: Argilla Science Dataset Setup & Upload
#
# This script provisions the **Panellinies Science** human evaluation dataset on Argilla:
# 1. Connects to the remote Argilla instance using project credentials.
# 2. Ensures the target workspace exists.
# 3. Ensures annotator accounts (`alexandra` and `alexandra_friends`) exist and are assigned to the workspace.
# 4. Sets up the Argilla evaluation schema with `min_submitted` task distribution.
# 5. Loads `human_evaluation_panellinies_science_subset.csv`, prioritizing app-corrected LaTeX edits (`Edited_*`).
# 6. Cleans and validates LaTeX / chemistry equations for KaTeX compatibility without stripping math syntax.
# 7. Enriches records with diagrams and images from the Hugging Face dataset (`ilsp/panellinies-exams-dataset`).
# 8. Pre-populates existing human grades for annotator `alexandra` when available.
# 9. Uploads all records to Argilla.

# %%
import os
import re
import json
import ast
import base64
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import UUID

import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login
import argilla as rg

# %% [markdown]
# ## 1. Configuration & Environment Setup

# %%
def find_project_root() -> Path:
    current = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current

PROJECT_ROOT = find_project_root()
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)

# Argilla Credentials
ARGILLA_API_URL = os.getenv("ARGILLA_API_URL")
ARGILLA_API_KEY = os.getenv("ARGILLA_API_KEY")

if not ARGILLA_API_URL or not ARGILLA_API_KEY:
    raise ValueError("Missing ARGILLA_API_URL or ARGILLA_API_KEY in environment variables.")

# Science Evaluation Parameters
WORKSPACE_NAME = os.getenv("POF_SCI_WORKSPACE_NAME", os.getenv("ARGILLA_POF_WORKSPACE", "pass-or-fail"))
DATASET_NAME = os.getenv("POF_SCI_DATASET_NAME", "pass-or-fail-science")

ANNOTATOR_1_NAME = os.getenv("POF_SCI_ANNOTATOR_1", "alexandra")
ANNOTATOR_1_PW = os.getenv("POF_SCI_ANNOTATOR_1_PASSWORD", "aaaaa")

ANNOTATOR_2_NAME = os.getenv("POF_SCI_ANNOTATOR_2", "alexandra_friends")
ANNOTATOR_2_PW = os.getenv("POF_SCI_ANNOTATOR_2_PASSWORD", "nnnn")

MIN_SUBMITTED = int(os.getenv("POF_SCI_MIN_SUBMITTED", "2"))
OVERWRITE = os.getenv("POF_SCI_OVERWRITE", "true").strip().lower() in ("true", "1", "yes")

CSV_PATH = Path(os.getenv("EVAL_SCI_CSV_PATH", PROJECT_ROOT / "scripts" / "human_evaluation_panellinies_science_subset.csv"))
HF_REPO = os.getenv("PANELLINIES_DATASET", "ilsp/panellinies-exams-dataset")
HF_TOKEN = os.getenv("HF_TOKEN")

print(f"Connecting to Argilla: {ARGILLA_API_URL}")
print(f"Target Workspace:     {WORKSPACE_NAME}")
print(f"Target Dataset:       {DATASET_NAME}")
print(f"Annotator 1:          {ANNOTATOR_1_NAME}")
print(f"Annotator 2:          {ANNOTATOR_2_NAME}")
print(f"Min Submitted:        {MIN_SUBMITTED}")
print(f"Overwrite Existing:   {OVERWRITE}")
print(f"Source CSV:           {CSV_PATH}")

if HF_TOKEN:
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"Hugging Face login warning: {e}")

client = rg.Argilla(api_url=ARGILLA_API_URL, api_key=ARGILLA_API_KEY)
print(f"Connected to Argilla (v{rg.__version__})")

# %% [markdown]
# ## 2. Workspace & Annotator User Provisioning

# %%
# Ensure workspace exists
workspace = client.workspaces(WORKSPACE_NAME)
if workspace is None:
    print(f"Workspace '{WORKSPACE_NAME}' not found. Creating...")
    workspace = rg.Workspace(name=WORKSPACE_NAME, client=client).create()
    print(f"Workspace '{workspace.name}' created.")
else:
    print(f"Workspace '{workspace.name}' found.")


def ensure_annotator_user(
    rg_client: rg.Argilla,
    target_workspace: rg.Workspace,
    username: str,
    password: str,
    role: str = "annotator",
) -> Optional[Any]:
    """Ensures an annotator user exists and is assigned to the workspace."""
    if not username:
        return None

    user = None
    try:
        user = rg_client.users(username)
    except Exception:
        try:
            for u in rg_client.users:
                if u.username == username:
                    user = u
                    break
        except Exception:
            pass

    if user is None:
        print(f"User '{username}' does not exist. Creating account...")
        try:
            user = rg.User(
                username=username,
                password=password,
                role=role,
                client=rg_client,
            ).create()
            print(f"User '{username}' created successfully.")
        except Exception as err:
            print(f"Notice: Could not create user '{username}' directly (endpoint may require admin): {err}")
    else:
        print(f"User '{username}' already exists.")

    if user is not None and target_workspace is not None:
        try:
            ws_users = getattr(target_workspace, "users", None) or []
            ws_user_names = [getattr(u, "username", str(u)) for u in ws_users]
            if username not in ws_user_names:
                print(f"Assigning '{username}' to workspace '{target_workspace.name}'...")
                if hasattr(target_workspace, "add_user"):
                    target_workspace.add_user(user)
                elif hasattr(user, "add_to_workspace"):
                    user.add_to_workspace(target_workspace)
        except Exception as err:
            print(f"Notice on workspace membership for '{username}': {err}")

    return user

# Ensure annotators
annotator_1_user = ensure_annotator_user(client, workspace, ANNOTATOR_1_NAME, ANNOTATOR_1_PW)
annotator_2_user = ensure_annotator_user(client, workspace, ANNOTATOR_2_NAME, ANNOTATOR_2_PW)

# %% [markdown]
# ## 3. LaTeX & Chemistry Normalization (KaTeX Compatible)

# %%
def fix_inline_math(s: str) -> str:
    """Ensure spaces around dollar signs adjacent to brackets or punctuation."""
    s = re.sub(r'(?<=[(\[{"\'])\$', ' $', s)
    s = re.sub(r'\$(?=[)\]}"\';])', '$ ', s)
    return s


def clean_science_latex(text: Any) -> str:
    """Normalize LaTeX and chemical equations so KaTeX renders without crashing in Argilla."""
    if not text or pd.isna(text):
        return ""
    text = str(text)

    # 1. Remove reasoning / thought blocks if present
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)

    # 2. Fix inline boundary spacing
    text = fix_inline_math(text)

    # 3. Normalize all math display/inline delimiters to standard single dollar signs
    text = text.replace('$$', '$')
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    text = text.replace(r'\[', '$').replace(r'\]', '$')

    # 4. Known common Panellinies LaTeX and OCR syntax bugs
    syntax_fixes = {
        "pK_{a}$)": "$pK_{a}$)",
        "pK_a$)": "$pK_a$)",
        r"\mathrm{M}$)": r"$\mathrm{M}$)",
        "Y_{2}$)": "$Y_{2}$)",
        "Y_{1}$ και Y_{2}$)": "$Y_{1}$ και $Y_{2}$)",
        "K_a = 10^{-5} $)": "$K_a = 10^{-5}$)",
        "aa$)": "aa)",
        "($aa$)": "(aa)",
        "($Aa": "(Aa",
        "Aa$)": "Aa)",
        r"\frac{[CH_3COO^-]}{[CH_3COOH]} = 1 $": r"$\frac{[CH_3COO^-]}{[CH_3COOH]} = 1$",
        r"\cdotpmin": r"\cdot \mathrm{min}",
        r"\cdotp": r"\cdot",
        "'$": "'",
        "$-": "-",
        "-$": "-",
        "< = >": r" \rightleftharpoons ",
        "<=>": r" \rightleftharpoons ",
        "⇌": r" \rightleftharpoons ",
        " - > ": r" \rightarrow ",
        " -> ": r" \rightarrow ",
        "→": r" \rightarrow ",
        "−": "-",
        "H_{3}$O+": "H_{3}O+",
        r"H\Delta$": r"H\Delta",
        r"\Delta^-$": r"\Delta^-",
        "SO_{3}$": "SO_{3}",
        "NO_{2}$": "NO_{2}",
        "SO_{2}$": "SO_{2}",
        "HO-$": "HO-",
        "C_{6}H_{5}-$": "C_{6}H_{5}-",
        "NO_{2}$ -": "NO_{2}-",
    }
    for bad, good in syntax_fixes.items():
        text = text.replace(bad, good)

    # 5. Convert chemical formulas (\ce{...}) to standard \mathrm{}
    def fix_chem(content: str) -> str:
        content = content.replace('$', '')
        content = re.sub(r'(?<=[A-Za-z)\]])(\d+)', r'_{\1}', content)
        content = re.sub(r'([+-]+)$', r'^{\1}', content)
        return f"\\mathrm{{{content}}}"

    text = re.sub(r'\\ce\s*{([^}]+)}', lambda m: fix_chem(m.group(1)), text)
    text = re.sub(r'\\ce\s*([A-Za-z0-9_+\-\^]+)', lambda m: fix_chem(m.group(1)), text)

    # 6. Ensure math segments don't contain unescaped line breaks
    text = re.sub(r'\$+', '$', text)
    parts = text.split('$')
    if len(parts) % 2 == 0:
        parts.append("")

    for i in range(1, len(parts), 2):
        parts[i] = parts[i].replace('\n', ' ').replace('\r', '')

    text = '$'.join(parts)

    # 7. Wrap naked LaTeX commands that appear outside $...$ in math mode
    new_parts = text.split('$')
    for i in range(0, len(new_parts), 2):
        new_parts[i] = re.sub(r'(\\mathrm\s*{[^{}]*})', r'$\1$', new_parts[i])
        new_parts[i] = re.sub(r'(\\d?frac\s*{[^{}]*}\s*{[^{}]*})', r'$\1$', new_parts[i])
        new_parts[i] = re.sub(r'(\\rightleftharpoons)', r'$\1$', new_parts[i])
        new_parts[i] = re.sub(r'(\\rightarrow)', r'$\1$', new_parts[i])
        new_parts[i] = re.sub(r'(\\Rightarrow)', r'$\1$', new_parts[i])
        new_parts[i] = re.sub(r'\b(X\^[aA])\b', r'$\1$', new_parts[i])

    text = '$'.join(new_parts)

    # 8. Clean excess delimiters and ensure balance
    text = re.sub(r'\$+', '$', text)
    text = text.replace('$$', '$')
    if text.count('$') % 2 != 0:
        text += '$'

    text = text.replace('\r', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# %% [markdown]
# ## 4. Dataset Schema & Guidelines Definition

# %%
GUIDELINES = """
### Οδηγίες Αξιολόγησης (Θετικές Επιστήμες - Πανελλαδικές)

Αξιολογείς την απάντηση του γλωσσικού μοντέλου (**Model answer**) στην ερώτηση (**Question**), συγκρίνοντάς τη με την πρότυπη απάντηση (**Reference answer**) και λαμβάνοντας υπόψη τυχόν σχετικό πλαίσιο/εκφώνηση (**Context**) ή διαγράμματα/εικόνες (**Images**).

#### Βαθμολογική Κλίμακα (Grade):
- **1.0**: Πλήρως ορθή και επιστημονικά άρτια απάντηση/επίλυση.
- **0.75**: Ορθή μεθοδολογική προσέγγιση με μικρές αριθμητικές ανακρίβειες ή επουσιώδη διατυπωτικά σφάλματα.
- **0.5**: Μερικώς ορθή απάντηση (π.χ. αναγνώριση μέρους των ζητουμένων, σωστή θεωρητική εκκίνηση χωρίς ολοκλήρωση).
- **0.25**: Ελάχιστα σωστά στοιχεία (π.χ. αναγραφή βασικού τύπου χωρίς ορθή εφαρμογή).
- **0.0**: Εντελώς λανθασμένη, παραπλανητική ή κενή απάντηση.

#### Αιτιολόγηση (Explanation):
- Προαιρετική σύντομη αιτιολόγηση του βαθμού στα Ελληνικά (συνιστάται σε περιπτώσεις μερικής βαθμολόγησης).
"""

dataset_settings = rg.Settings(
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
    ],
    guidelines=GUIDELINES.strip(),
    distribution=rg.TaskDistribution(min_submitted=MIN_SUBMITTED),
)

# %% [markdown]
# ## 5. Dataset Creation / Overwrite Handling

# %%
existing_dataset = client.datasets(name=DATASET_NAME, workspace=WORKSPACE_NAME)
if existing_dataset is not None:
    if OVERWRITE:
        print(f"POF_SCI_OVERWRITE is True. Deleting existing dataset '{DATASET_NAME}'...")
        existing_dataset.delete()
        existing_dataset = None
        print("Existing dataset deleted.")
    else:
        print(f"Dataset '{DATASET_NAME}' already exists and OVERWRITE is False. Reusing existing dataset.")

if existing_dataset is None:
    print(f"Creating new Argilla dataset '{DATASET_NAME}'...")
    dataset = rg.Dataset(
        name=DATASET_NAME,
        workspace=WORKSPACE_NAME,
        settings=dataset_settings,
        client=client,
    ).create()
    print(f"Dataset '{dataset.name}' created successfully.")
else:
    dataset = existing_dataset

# %% [markdown]
# ## 6. Data Parsing & HF Enrichment

# %%
def images_to_markdown(images_list: Any) -> str:
    """Convert Hugging Face PIL images to embedded base64 markdown img tags."""
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
        except Exception:
            pass
    return "\n\n".join(md_elements)


def parse_dict_field(val: Any) -> Dict[str, Any]:
    if pd.isna(val):
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return json.loads(s)
            except Exception:
                try:
                    return ast.literal_eval(s)
                except Exception:
                    pass
    return {}


def unpack_field(val: Any, preferred_key: Optional[str] = None) -> str:
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

# Load local Science CSV
if not CSV_PATH.exists():
    raise FileNotFoundError(f"Science CSV not found at: {CSV_PATH}")

print(f"Loading Science evaluation CSV from: {CSV_PATH}")
df_subset = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df_subset)} records from CSV.")

# Load Hugging Face reference dataset for image and context enrichment
print(f"Fetching Hugging Face reference dataset '{HF_REPO}'...")
try:
    hf_dataset_dict = load_dataset(HF_REPO)
    hf_index = {item["id"]: item for split in hf_dataset_dict.keys() for item in hf_dataset_dict[split]}
    print(f"Indexed {len(hf_index)} reference items from Hugging Face.")
except Exception as e:
    print(f"Notice: Could not fetch '{HF_REPO}' ({e}). Relying on CSV contents directly.")
    hf_index = {}

# %% [markdown]
# ## 7. Record Construction & Upload

# %%
alexandra_user_id = getattr(annotator_1_user, "id", None) if annotator_1_user else None

records: List[rg.Record] = []
pre_annotated_count = 0
images_attached_count = 0

for _, row in df_subset.iterrows():
    qid = str(row["Question_ID"]).strip()
    hf_item = hf_index.get(qid, {})

    # 1. Unpack Raw CSV content
    input_q_dict = parse_dict_field(row.get("Input_Question"))
    csv_question = input_q_dict.get("question", "") if input_q_dict else str(row.get("Input_Question") or "").strip()
    csv_context = input_q_dict.get("context", "") if input_q_dict else ""
    csv_ans = unpack_field(row.get("Reference_Target"), preferred_key="reference")
    model_ans = unpack_field(row.get("Model_Answer"), preferred_key="answer")

    # 2. Check for human-corrected edits in CSV (prioritize Edited_* if populated)
    edited_q = str(row.get("Edited_Question") or "").strip()
    edited_ref = str(row.get("Edited_Reference") or "").strip()
    edited_model = str(row.get("Edited_Model_Answer") or "").strip()

    # 3. Resolve canonical text with HF enrichment
    if edited_q:
        question_text = clean_science_latex(edited_q)
    elif hf_item and hf_item.get("question"):
        question_text = clean_science_latex(hf_item.get("question"))
    else:
        question_text = clean_science_latex(csv_question)

    if hf_item and hf_item.get("input"):
        context_input = clean_science_latex(hf_item.get("input"))
    else:
        context_input = clean_science_latex(csv_context)

    images_rendered = ""
    image_desc = ""
    if hf_item:
        images_rendered = images_to_markdown(hf_item.get("images", []))
        if images_rendered:
            images_attached_count += 1
        raw_desc = hf_item.get("image_description") or hf_item.get("image_transcription") or ""
        image_desc = clean_science_latex(raw_desc)

    if edited_ref:
        ref_ans = clean_science_latex(edited_ref)
    elif csv_ans:
        ref_ans = clean_science_latex(csv_ans)
    elif hf_item and hf_item.get("answer_text"):
        ref_ans = clean_science_latex(hf_item.get("answer_text"))
    else:
        ref_ans = ""

    if edited_model:
        final_model_ans = clean_science_latex(edited_model)
    else:
        final_model_ans = clean_science_latex(model_ans)

    model_name = str(row["Model"]).strip() if pd.notna(row["Model"]) else ""
    record_id = f"{qid}__{model_name}"

    metadata = {
        "subject": str(row["Subject"]) if pd.notna(row["Subject"]) else "",
        "question_id": qid,
        "model_name": model_name,
    }

    # 4. Attach existing human evaluation for alexandra if present
    responses = []
    human_grade_val = row.get("Human_Grade")
    if alexandra_user_id and pd.notna(human_grade_val):
        try:
            uid = UUID(str(alexandra_user_id))
            grade_float = float(human_grade_val)
            closest_label = min(["0.0", "0.25", "0.5", "0.75", "1.0"], key=lambda x: abs(float(x) - grade_float))
            explanation_val = str(row.get("Explanation") or "").strip()

            responses.append(
                rg.Response(
                    question_name="grade",
                    value=closest_label,
                    user_id=uid,
                    status="submitted",
                )
            )
            if explanation_val:
                responses.append(
                    rg.Response(
                        question_name="explanation",
                        value=explanation_val,
                        user_id=uid,
                        status="submitted",
                    )
                )
            pre_annotated_count += 1
        except Exception:
            pass

    record_kwargs = {
        "id": record_id,
        "fields": {
            "question_id_field": qid,
            "input": context_input,
            "question": question_text,
            "images": images_rendered,
            "image_description": image_desc,
            "reference_answer": ref_ans,
            "answer": final_model_ans,
        },
        "metadata": metadata,
    }
    if responses:
        record_kwargs["responses"] = responses

    records.append(rg.Record(**record_kwargs))

print(f"\nPrepared {len(records)} records for Argilla:")
print(f"- Records with embedded diagrams: {images_attached_count}")
print(f"- Records pre-populated with existing evaluations for '{ANNOTATOR_1_NAME}': {pre_annotated_count}")

# %% [markdown]
# ## 8. Log Records to Argilla

# %%
print(f"\nLogging records to Argilla dataset '{DATASET_NAME}' (batch size: 50)...")
BATCH_SIZE = 50
for i in range(0, len(records), BATCH_SIZE):
    batch = records[i : i + BATCH_SIZE]
    dataset.records.log(batch)
    print(f"  Logged records [{i + len(batch)}/{len(records)}]")

print(f"\nSuccessfully uploaded {len(records)} records to '{DATASET_NAME}' in workspace '{WORKSPACE_NAME}'.")
print(f"Annotator '{ANNOTATOR_1_NAME}' can now log in at: {ARGILLA_API_URL}")

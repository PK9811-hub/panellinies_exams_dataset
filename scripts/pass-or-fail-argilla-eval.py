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

# # %%
# import os
# import json
# import ast
# import base64
# from io import BytesIO
# from pathlib import Path
# import pandas as pd
# from dotenv import load_dotenv
# from datasets import load_dataset
# from huggingface_hub import login
# import argilla as rg

# # --- Φόρτωση Μεταβλητών Περιβάλλοντος ---
# load_dotenv()

# argilla_api_url = os.getenv("ARGILLA_API_URL")
# argilla_api_key = os.getenv("ARGILLA_API_KEY")
# workspace_name = os.getenv("ARGILLA_POF_WORKSPACE", "pass-or-fail")
# hf_token = os.getenv("HF_TOKEN")

# if not argilla_api_url or not argilla_api_key:
#     print("\n[ΣΦΑΛΜΑ] Δεν βρέθηκαν τα ARGILLA_API_URL ή ARGILLA_API_KEY!")
#     exit(1)

# # --- Ρυθμίσεις ---
# RECREATE_DATASET = True
# MIN_SUBMITTED = 1

# # --- ΑΥΣΤΗΡΗ ΟΝΟΜΑΤΟΔΟΣΙΑ ΓΙΑ ΑΝΘΡΩΠΙΣΤΙΚΑ ---
# dataset_name = "pass-or-fail-hum"
# csv_file = "human_evaluation_panellinies_hum_subset.csv"
# hf_repo = os.getenv("PANELLINIES_DATASET", "ilsp/panellinies-exams-dataset")

# if hf_token:
#     try:
#         login(token=hf_token, add_to_git_credential=False)
#         print("Logged in to Hugging Face Hub.")
#     except Exception as e:
#         pass

# client = rg.Argilla(api_url=argilla_api_url, api_key=argilla_api_key)
# print(f"Argilla Client connected. Argilla version: {rg.__version__}")

# # --- 1. Διαχείριση Workspace ---
# workspace = client.workspaces(workspace_name)
# if workspace is None:
#     print(f"Workspace '{workspace_name}' does not exist. Creating...")
#     workspace = rg.Workspace(name=workspace_name, client=client).create()
# else:
#     print(f"Workspace '{workspace.name}' found.")

# # Η ΔΙΑΧΕΙΡΙΣΗ ΧΡΗΣΤΩΝ ΑΦΑΙΡΕΘΗΚΕ ΛΟΓΩ 403 FORBIDDEN

# # --- 2. Σχήμα Dataset ---
# guidelines = """
# ### Οδηγίες Αξιολόγησης (Ανθρωπιστικά - Πανελλαδικές)
# Αξιολογείς μια υποβληθείσα απάντηση (**Model answer**) σε μια ερώτηση (**Question**), συγκρίνοντάς τη με την πρότυπη απάντηση (**Reference answer**) και λαμβάνοντας υπόψη το πλαίσιο/κείμενο (**Context**).

# #### Βαθμολογική Κλίμακα (Grade):
# - **1.0**: Πλήρως ορθή και ολοκληρωμένη απάντηση / μετάφραση / σχολιασμός.
# - **0.75**: Ορθή προσέγγιση με μικρές παραλείψεις ή επουσιώδη διατυπωτικά σφάλματα.
# - **0.5**: Μερικώς ορθή απάντηση (π.χ. αναγνώριση των μισών ζητούμενων).
# - **0.25**: Ελάχιστα σωστά στοιχεία (π.χ. σωστή ρίζα λέξης, αλλά λάθος συντακτικό/γραμματική).
# - **0.0**: Εντελώς λανθασμένη, άσχετη ή κενή απάντηση.

# #### Αιτιολόγηση (Explanation):
# - Σύντομη αιτιολόγηση του βαθμού στα Ελληνικά (απαραίτητη σε περιπτώσεις μερικής βαθμολόγησης).
# """

# settings = rg.Settings(
#     fields=[
#         rg.TextField(name="question_id_field", title="Question ID", use_markdown=False),
#         rg.TextField(name="input", title="Context", use_markdown=True, required=False),
#         rg.TextField(name="question", title="Question", use_markdown=True),
#         rg.TextField(name="images", title="Images", use_markdown=True, required=False),
#         rg.TextField(name="image_description", title="Image description", use_markdown=True, required=False),
#         rg.TextField(name="reference_answer", title="Reference answer", use_markdown=True),
#         rg.TextField(name="answer", title="Model answer", use_markdown=True),
#     ],
#     questions=[
#         rg.LabelQuestion(name="grade", title="Grade", labels=["0.0", "0.25", "0.5", "0.75", "1.0"], required=True),
#         rg.TextQuestion(name="explanation", title="Explanation", use_markdown=True, required=False),
#     ],
#     metadata=[
#         rg.TermsMetadataProperty(name="subject", title="Subject", visible_for_annotators=True),
#         rg.TermsMetadataProperty(name="question_id", title="Question ID", visible_for_annotators=True),
#         rg.TermsMetadataProperty(name="model_name", title="Model", visible_for_annotators=False),
#     ],
#     guidelines=guidelines.strip(),
#     distribution=rg.TaskDistribution(min_submitted=MIN_SUBMITTED),
# )

# # --- 3. Φόρτωση Δεδομένων & Εμπλουτισμός ---
# def images_to_markdown(images_list):
#     if not images_list: return ""
#     md_elements = []
#     for idx, img in enumerate(images_list):
#         if img is None: continue
#         try:
#             if hasattr(img, "save"):
#                 buf = BytesIO()
#                 img.save(buf, format="PNG")
#                 b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
#                 md_elements.append(f'<img src="data:image/png;base64,{b64_str}" alt="Diagram {idx+1}" style="max-width: 100%; height: auto;" />')
#             elif isinstance(img, str) and img.strip():
#                 md_elements.append(f"![Diagram {idx+1}]({img})")
#         except Exception:
#             pass
#     return "\n\n".join(md_elements)

# def parse_dict_field(val):
#     if pd.isna(val): return {}
#     if isinstance(val, dict): return val
#     if isinstance(val, str):
#         s = val.strip()
#         if s.startswith("{") and s.endswith("}"):
#             try: return json.loads(s)
#             except Exception:
#                 try: return ast.literal_eval(s)
#                 except Exception: pass
#     return {}

# def unpack_field(val, preferred_key=None):
#     parsed = parse_dict_field(val)
#     if parsed:
#         if preferred_key and preferred_key in parsed: return str(parsed[preferred_key]).strip()
#         for k in ["answer", "reference", "question", "context", "text", "content"]:
#             if k in parsed: return str(parsed[k]).strip()
#         if len(parsed) == 1: return str(next(iter(parsed.values()))).strip()
#     return str(val).strip() if pd.notna(val) else ""

# print(f"Loading local CSV: {csv_file}")
# df_subset = pd.read_csv(csv_file)
# print(f"Loaded {len(df_subset)} records.")

# print(f"Loading Hugging Face dataset '{hf_repo}'...")
# hf_dataset_dict = load_dataset(hf_repo)

# hf_index = {item["id"]: item for split in hf_dataset_dict.keys() for item in hf_dataset_dict[split]}

# existing_dataset = client.datasets(name=dataset_name, workspace=workspace_name)
# if existing_dataset and RECREATE_DATASET:
#     existing_dataset.delete()
#     existing_dataset = None

# if existing_dataset:
#     dataset = existing_dataset
# else:
#     dataset = rg.Dataset(name=dataset_name, workspace=workspace_name, settings=settings, client=client).create()

# records = []
# for _, row in df_subset.iterrows():
#     qid = str(row["Question_ID"])
#     hf_item = hf_index.get(qid, {})

#     input_q_dict = parse_dict_field(row.get("Input_Question"))
#     csv_question = input_q_dict.get("question", "") if input_q_dict else str(row.get("Input_Question") or "").strip()
#     csv_context = input_q_dict.get("context", "") if input_q_dict else ""
#     csv_ans = unpack_field(row.get("Reference_Target"), preferred_key="reference")
#     model_ans = unpack_field(row.get("Model_Answer"), preferred_key="answer")

#     if hf_item:
#         question_text = str(hf_item.get("question") or csv_question)
#         context_input = str(hf_item.get("input") or csv_context)
#         images_rendered = images_to_markdown(hf_item.get("images", []))
#         image_desc = str(hf_item.get("image_description") or hf_item.get("image_transcription") or "")
        
#         hf_ans = str(hf_item.get("answer_text") or "").strip()
#         if hf_ans and csv_ans and hf_ans != csv_ans:
#             ref_ans = f"**Answer Key:** {hf_ans}\n\n**Detailed Solution:**\n{csv_ans}"
#         else:
#             ref_ans = hf_ans or csv_ans
#     else:
#         question_text, context_input, images_rendered, image_desc, ref_ans = csv_question, csv_context, "", "", csv_ans

#     metadata = {
#         "subject": str(row["Subject"]) if pd.notna(row["Subject"]) else "",
#         "model_name": str(row["Model"]) if pd.notna(row["Model"]) else "",
#         "question_id": qid,
#     }

#     records.append(rg.Record(
#         fields={
#             "question_id_field": qid,
#             "input": context_input,
#             "question": question_text,
#             "images": images_rendered,
#             "image_description": image_desc,
#             "reference_answer": ref_ans,
#             "answer": model_ans,
#         },
#         metadata=metadata,
#     ))

# print(f"Logging {len(records)} records to Argilla dataset '{dataset_name}'...")
# dataset.records.log(records)
# print("Upload complete!")


import os
import json
import ast
import base64
import re
from io import BytesIO
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import login
import argilla as rg

# --- Φόρτωση Μεταβλητών Περιβάλλοντος ---
load_dotenv()

argilla_api_url = os.getenv("ARGILLA_API_URL")
argilla_api_key = os.getenv("ARGILLA_API_KEY")
workspace_name = os.getenv("ARGILLA_POF_WORKSPACE", "pass-or-fail")
hf_token = os.getenv("HF_TOKEN")

if not argilla_api_url or not argilla_api_key:
    print("\n[ΣΦΑΛΜΑ] Δεν βρέθηκαν τα ARGILLA_API_URL ή ARGILLA_API_KEY!")
    exit(1)

# --- Ρυθμίσεις ---
RECREATE_DATASET = True
MIN_SUBMITTED = 1

# --- ΑΥΣΤΗΡΗ ΟΝΟΜΑΤΟΔΟΣΙΑ ΓΙΑ ΘΕΤΙΚΕΣ ΕΠΙΣΤΗΜΕΣ ---
dataset_name = "pass-or-fail-science"
csv_file = "human_evaluation_panellinies_science_subset.csv"
hf_repo = os.getenv("PANELLINIES_DATASET", "ilsp/panellinies-exams-dataset")

if hf_token:
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("Logged in to Hugging Face Hub.")
    except Exception as e:
        pass

client = rg.Argilla(api_url=argilla_api_url, api_key=argilla_api_key)
print(f"Argilla Client connected. Argilla version: {rg.__version__}")

# --- 1. Διαχείριση Workspace ---
workspace = client.workspaces(workspace_name)
if workspace is None:
    print(f"Workspace '{workspace_name}' does not exist. Creating...")
    workspace = rg.Workspace(name=workspace_name, client=client).create()
else:
    print(f"Workspace '{workspace.name}' found.")

# Η ΔΙΑΧΕΙΡΙΣΗ ΧΡΗΣΤΩΝ ΑΦΑΙΡΕΘΗΚΕ ΛΟΓΩ 403 FORBIDDEN

# --- 2. ΦΙΛΤΡΟ ΚΑΘΑΡΙΣΜΟΥ LATEX ΣΕ ΑΠΛΟ ΚΕΙΜΕΝΟ ---
def replace_fractions(text):
    for cmd in ['\\frac', '\\dfrac']:
        safety = 0
        while cmd in text and safety < 100:
            safety += 1
            start_idx = text.find(cmd)
            brace1_start = text.find('{', start_idx)
            if brace1_start == -1 or brace1_start > start_idx + 8:
                text = text.replace(cmd, cmd.replace('\\', ''), 1)
                continue
            
            brace1_end = -1
            depth = 0
            for i in range(brace1_start, len(text)):
                if text[i] == '{': depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        brace1_end = i
                        break
            if brace1_end == -1: 
                text = text.replace(cmd, cmd.replace('\\', ''), 1)
                continue
            
            num = text[brace1_start+1:brace1_end]
            
            brace2_start = text.find('{', brace1_end + 1)
            if brace2_start == -1 or brace2_start > brace1_end + 3:
                text = text.replace(cmd, cmd.replace('\\', ''), 1)
                continue
            
            brace2_end = -1
            depth = 0
            for i in range(brace2_start, len(text)):
                if text[i] == '{': depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        brace2_end = i
                        break
            if brace2_end == -1: 
                text = text.replace(cmd, cmd.replace('\\', ''), 1)
                continue
            
            den = text[brace2_start+1:brace2_end]
            full_match = text[start_idx:brace2_end+1]
            text = text.replace(full_match, f"({num})/({den})")
    return text

def replace_sqrt(text):
    safety = 0
    while '\\sqrt' in text and safety < 100:
        safety += 1
        start_idx = text.find('\\sqrt')
        brace1_start = text.find('{', start_idx)
        if brace1_start == -1 or brace1_start > start_idx + 6:
            text = text.replace('\\sqrt', 'sqrt', 1)
            continue
            
        brace1_end = -1
        depth = 0
        for i in range(brace1_start, len(text)):
            if text[i] == '{': depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    brace1_end = i
                    break
        if brace1_end == -1: 
            text = text.replace('\\sqrt', 'sqrt', 1)
            continue
            
        content = text[brace1_start+1:brace1_end]
        full_match = text[start_idx:brace1_end+1]
        text = text.replace(full_match, f"√({content})")
    return text

def flatten_latex_for_argilla(text):
    if not text or pd.isna(text): return ""
    text = str(text)

    # Καθαρισμός <think>
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Αφαίρεση τοξικών εντολών KaTeX (Κρατάμε μόνο το περιεχόμενό τους)
    text = re.sub(r'\\ce{([^}]+)}', r'\1', text)
    text = re.sub(r'\\ce([A-Za-z0-9_+\-]+)', r'\1', text)
    text = re.sub(r'\\mathrm{([^}]+)}', r'\1', text)
    text = re.sub(r'\\text{([^}]+)}', r'\1', text)
    
    # Μετατροπή κλασμάτων και ριζών
    text = replace_fractions(text)
    text = replace_sqrt(text)
    
    # Unicode αντικαταστάσεις
    replacements = {
        r'\Rightarrow': '=>',
        r'\rightarrow': '->',
        r'\rightleftharpoons': '<=>',
        r'\cdot': '*',
        r'\Delta': 'Δ',
        r'\alpha': 'α',
        r'\beta': 'β',
        r'\gamma': 'γ',
        r'\pi': 'π',
        '< = >': '<=>',
        ' - > ': '->',
        ' -> ': ' -> '
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
        
    # ΑΠΕΝΕΡΓΟΠΟΙΗΣΗ ΤΟΥ LATEX: Σβήνουμε όλα τα σύμβολα
    text = text.replace(r'\(', '').replace(r'\)', '')
    text = text.replace(r'\[', '').replace(r'\]', '')
    text = text.replace('$', '')
    
    # Διορθώσεις λαθών HF dataset που άφηναν ανοιχτά $
    text = text.replace("pK_{a})", "pKa)")
    text = text.replace("pK_a)", "pKa)")
    text = text.replace(r"\mathrm{M})", "M)")
    text = text.replace("'", "'").replace("-", "-")
    
    # Escaping του % 
    text = re.sub(r'(?<!\\)%', r'\%', text)
        
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- 3. Σχήμα Dataset ---
guidelines = """
### Οδηγίες Αξιολόγησης (Θετικές Επιστήμες - Πανελλαδικές)
Αξιολογείς μια υποβληθείσα απάντηση (**Model answer**) σε μια ερώτηση (**Question**), συγκρίνοντάς τη με την πρότυπη απάντηση (**Reference answer**) και λαμβάνοντας υπόψη το πλαίσιο/δεδομένα (**Context**) και τυχόν εικόνες/σχήματα (**Images**).

#### Βαθμολογική Κλίμακα (Grade):
- **1.0**: Πλήρως ορθή και ολοκληρωμένη απάντηση / επίλυση.
- **0.75**: Ορθή προσέγγιση / μεθοδολογία, αλλά με μικρά αριθμητικά λάθη ή επουσιώδεις παραλείψεις.
- **0.5**: Μερικώς ορθή απάντηση (π.χ. σωστή η μισή άσκηση ή σωστός τύπος με λάθος εφαρμογή).
- **0.25**: Ελάχιστα σωστά στοιχεία (π.χ. απλή αναφορά του σωστού τύπου χωρίς καμία λογική συνέχεια).
- **0.0**: Εντελώς λανθασμένη, άσχετη ή κενή απάντηση.

#### Αιτιολόγηση (Explanation):
- Σύντομη αιτιολόγηση του βαθμού στα Ελληνικά (απαραίτητη σε περιπτώσεις μερικής βαθμολόγησης).
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
        rg.LabelQuestion(name="grade", title="Grade", labels=["0.0", "0.25", "0.5", "0.75", "1.0"], required=True),
        rg.TextQuestion(name="explanation", title="Explanation", use_markdown=True, required=False),
    ],
    metadata=[
        rg.TermsMetadataProperty(name="subject", title="Subject", visible_for_annotators=True),
        rg.TermsMetadataProperty(name="question_id", title="Question ID", visible_for_annotators=True),
        rg.TermsMetadataProperty(name="model_name", title="Model", visible_for_annotators=False),
    ],
    guidelines=guidelines.strip(),
    distribution=rg.TaskDistribution(min_submitted=MIN_SUBMITTED),
)

# --- 4. Φόρτωση Δεδομένων & Εμπλουτισμός ---
def images_to_markdown(images_list):
    if not images_list: return ""
    md_elements = []
    for idx, img in enumerate(images_list):
        if img is None: continue
        try:
            if hasattr(img, "save"):
                buf = BytesIO()
                img.save(buf, format="PNG")
                b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
                md_elements.append(f'<img src="data:image/png;base64,{b64_str}" alt="Diagram {idx+1}" style="max-width: 100%; height: auto;" />')
            elif isinstance(img, str) and img.strip():
                md_elements.append(f"![Diagram {idx+1}]({img})")
        except Exception:
            pass
    return "\n\n".join(md_elements)

def parse_dict_field(val):
    if pd.isna(val): return {}
    if isinstance(val, dict): return val
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{") and s.endswith("}"):
            try: return json.loads(s)
            except Exception:
                try: return ast.literal_eval(s)
                except Exception: pass
    return {}

def unpack_field(val, preferred_key=None):
    parsed = parse_dict_field(val)
    if parsed:
        if preferred_key and preferred_key in parsed: return str(parsed[preferred_key]).strip()
        for k in ["answer", "reference", "question", "context", "text", "content"]:
            if k in parsed: return str(parsed[k]).strip()
        if len(parsed) == 1: return str(next(iter(parsed.values()))).strip()
    return str(val).strip() if pd.notna(val) else ""

print(f"Loading local CSV: {csv_file}")
df_subset = pd.read_csv(csv_file)
print(f"Loaded {len(df_subset)} records.")

print(f"Loading Hugging Face dataset '{hf_repo}'...")
hf_dataset_dict = load_dataset(hf_repo)

hf_index = {item["id"]: item for split in hf_dataset_dict.keys() for item in hf_dataset_dict[split]}

existing_dataset = client.datasets(name=dataset_name, workspace=workspace_name)
if existing_dataset and RECREATE_DATASET:
    existing_dataset.delete()
    existing_dataset = None

if existing_dataset:
    dataset = existing_dataset
else:
    dataset = rg.Dataset(name=dataset_name, workspace=workspace_name, settings=settings, client=client).create()

records = []
for _, row in df_subset.iterrows():
    qid = str(row["Question_ID"])
    hf_item = hf_index.get(qid, {})

    input_q_dict = parse_dict_field(row.get("Input_Question"))
    csv_question = input_q_dict.get("question", "") if input_q_dict else str(row.get("Input_Question") or "").strip()
    csv_context = input_q_dict.get("context", "") if input_q_dict else ""
    csv_ans = unpack_field(row.get("Reference_Target"), preferred_key="reference")
    model_ans = unpack_field(row.get("Model_Answer"), preferred_key="answer")

    if hf_item:
        question_text = flatten_latex_for_argilla(str(hf_item.get("question") or csv_question))
        context_input = flatten_latex_for_argilla(str(hf_item.get("input") or csv_context))
        images_rendered = images_to_markdown(hf_item.get("images", []))
        image_desc = flatten_latex_for_argilla(str(hf_item.get("image_description") or hf_item.get("image_transcription") or ""))
        
        hf_ans = flatten_latex_for_argilla(str(hf_item.get("answer_text") or ""))
        csv_ans_clean = flatten_latex_for_argilla(csv_ans)
        
        if hf_ans and csv_ans_clean and hf_ans != csv_ans_clean:
            ref_ans = f"**Answer Key:** {hf_ans}\n\n**Detailed Solution:**\n{csv_ans_clean}"
        else:
            ref_ans = hf_ans or csv_ans_clean
    else:
        question_text = flatten_latex_for_argilla(csv_question)
        context_input = flatten_latex_for_argilla(csv_context)
        images_rendered = ""
        image_desc = ""
        ref_ans = flatten_latex_for_argilla(csv_ans)
        
    model_ans = flatten_latex_for_argilla(model_ans)

    metadata = {
        "subject": str(row["Subject"]) if pd.notna(row["Subject"]) else "",
        "model_name": str(row["Model"]) if pd.notna(row["Model"]) else "",
        "question_id": qid,
    }

    records.append(rg.Record(
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
    ))

print(f"Logging {len(records)} records to Argilla dataset '{dataset_name}'...")
dataset.records.log(records)
print("Upload complete!")
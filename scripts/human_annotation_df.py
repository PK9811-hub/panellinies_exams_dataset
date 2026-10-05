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
# 4. Defines the Argilla dataset schema (`pass-or-fail-science`) tailored for human evaluation.
# 5. Fetches the human evaluation subset CSV and enriches each record.
# 6. Flattens LaTeX math and chemistry to plain text to prevent Argilla KaTeX crashes.
# 7. Uploads the enriched records to Argilla.

# %%
# import os
# import json
# import ast
# import base64
# import re
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

# # --- ΑΥΣΤΗΡΗ ΟΝΟΜΑΤΟΔΟΣΙΑ ΓΙΑ ΘΕΤΙΚΕΣ ΕΠΙΣΤΗΜΕΣ ---
# dataset_name = "pass-or-fail-science"
# csv_file = "human_evaluation_panellinies_science_subset.csv"
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

# # --- 2. ΦΙΛΤΡΟ ΚΑΘΑΡΙΣΜΟΥ LATEX ΣΕ ΑΠΛΟ ΚΕΙΜΕΝΟ ---
# def replace_fractions(text):
#     for cmd in ['\\frac', '\\dfrac']:
#         safety = 0
#         while cmd in text and safety < 100:
#             safety += 1
#             start_idx = text.find(cmd)
#             brace1_start = text.find('{', start_idx)
#             if brace1_start == -1 or brace1_start > start_idx + 8:
#                 text = text.replace(cmd, cmd.replace('\\', ''), 1)
#                 continue
            
#             brace1_end = -1
#             depth = 0
#             for i in range(brace1_start, len(text)):
#                 if text[i] == '{': depth += 1
#                 elif text[i] == '}':
#                     depth -= 1
#                     if depth == 0:
#                         brace1_end = i
#                         break
#             if brace1_end == -1: 
#                 text = text.replace(cmd, cmd.replace('\\', ''), 1)
#                 continue
            
#             num = text[brace1_start+1:brace1_end]
            
#             brace2_start = text.find('{', brace1_end + 1)
#             if brace2_start == -1 or brace2_start > brace1_end + 3:
#                 text = text.replace(cmd, cmd.replace('\\', ''), 1)
#                 continue
            
#             brace2_end = -1
#             depth = 0
#             for i in range(brace2_start, len(text)):
#                 if text[i] == '{': depth += 1
#                 elif text[i] == '}':
#                     depth -= 1
#                     if depth == 0:
#                         brace2_end = i
#                         break
#             if brace2_end == -1: 
#                 text = text.replace(cmd, cmd.replace('\\', ''), 1)
#                 continue
            
#             den = text[brace2_start+1:brace2_end]
#             full_match = text[start_idx:brace2_end+1]
#             text = text.replace(full_match, f"({num})/({den})")
#     return text

# def replace_sqrt(text):
#     safety = 0
#     while '\\sqrt' in text and safety < 100:
#         safety += 1
#         start_idx = text.find('\\sqrt')
#         brace1_start = text.find('{', start_idx)
#         if brace1_start == -1 or brace1_start > start_idx + 6:
#             text = text.replace('\\sqrt', 'sqrt', 1)
#             continue
            
#         brace1_end = -1
#         depth = 0
#         for i in range(brace1_start, len(text)):
#             if text[i] == '{': depth += 1
#             elif text[i] == '}':
#                 depth -= 1
#                 if depth == 0:
#                     brace1_end = i
#                     break
#         if brace1_end == -1: 
#             text = text.replace('\\sqrt', 'sqrt', 1)
#             continue
            
#         content = text[brace1_start+1:brace1_end]
#         full_match = text[start_idx:brace1_end+1]
#         text = text.replace(full_match, f"√({content})")
#     return text

# def flatten_latex_for_argilla(text):
#     if not text or pd.isna(text): return ""
#     text = str(text)

#     # Καθαρισμός <think>
#     text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    
#     # Αφαίρεση τοξικών εντολών KaTeX (Κρατάμε μόνο το περιεχόμενό τους)
#     text = re.sub(r'\\ce{([^}]+)}', r'\1', text)
#     text = re.sub(r'\\ce([A-Za-z0-9_+\-]+)', r'\1', text)
#     text = re.sub(r'\\mathrm{([^}]+)}', r'\1', text)
#     text = re.sub(r'\\text{([^}]+)}', r'\1', text)
    
#     # Μετατροπή κλασμάτων και ριζών
#     text = replace_fractions(text)
#     text = replace_sqrt(text)
    
#     # Unicode αντικαταστάσεις
#     replacements = {
#         r'\Rightarrow': '=>',
#         r'\rightarrow': '->',
#         r'\rightleftharpoons': '<=>',
#         r'\cdot': '*',
#         r'\Delta': 'Δ',
#         r'\alpha': 'α',
#         r'\beta': 'β',
#         r'\gamma': 'γ',
#         r'\pi': 'π',
#         '< = >': '<=>',
#         ' - > ': '->',
#         ' -> ': ' -> '
#     }
#     for k, v in replacements.items():
#         text = text.replace(k, v)
        
#     # ΑΠΕΝΕΡΓΟΠΟΙΗΣΗ ΤΟΥ LATEX: Σβήνουμε όλα τα σύμβολα
#     text = text.replace(r'\(', '').replace(r'\)', '')
#     text = text.replace(r'\[', '').replace(r'\]', '')
#     text = text.replace('$', '')
    
#     # Διορθώσεις λαθών HF dataset που άφηναν ανοιχτά $
#     text = text.replace("pK_{a})", "pKa)")
#     text = text.replace("pK_a)", "pKa)")
#     text = text.replace(r"\mathrm{M})", "M)")
#     text = text.replace("'", "'").replace("-", "-")
    
#     # Escaping του % 
#     text = re.sub(r'(?<!\\)%', r'\%', text)
        
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

# # --- 3. Σχήμα Dataset ---
# guidelines = """
# ### Οδηγίες Αξιολόγησης (Θετικές Επιστήμες - Πανελλαδικές)
# Αξιολογείς μια υποβληθείσα απάντηση (**Model answer**) σε μια ερώτηση (**Question**), συγκρίνοντάς τη με την πρότυπη απάντηση (**Reference answer**) και λαμβάνοντας υπόψη το πλαίσιο/δεδομένα (**Context**) και τυχόν εικόνες/σχήματα (**Images**).

# #### Βαθμολογική Κλίμακα (Grade):
# - **1.0**: Πλήρως ορθή και ολοκληρωμένη απάντηση / επίλυση.
# - **0.75**: Ορθή προσέγγιση / μεθοδολογία, αλλά με μικρά αριθμητικά λάθη ή επουσιώδεις παραλείψεις.
# - **0.5**: Μερικώς ορθή απάντηση (π.χ. σωστή η μισή άσκηση ή σωστός τύπος με λάθος εφαρμογή).
# - **0.25**: Ελάχιστα σωστά στοιχεία (π.χ. απλή αναφορά του σωστού τύπου χωρίς καμία λογική συνέχεια).
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

# # --- 4. Φόρτωση Δεδομένων & Εμπλουτισμός ---
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
#         question_text = flatten_latex_for_argilla(str(hf_item.get("question") or csv_question))
#         context_input = flatten_latex_for_argilla(str(hf_item.get("input") or csv_context))
#         images_rendered = images_to_markdown(hf_item.get("images", []))
#         image_desc = flatten_latex_for_argilla(str(hf_item.get("image_description") or hf_item.get("image_transcription") or ""))
        
#         hf_ans = flatten_latex_for_argilla(str(hf_item.get("answer_text") or ""))
#         csv_ans_clean = flatten_latex_for_argilla(csv_ans)
        
#         if hf_ans and csv_ans_clean and hf_ans != csv_ans_clean:
#             ref_ans = f"**Answer Key:** {hf_ans}\n\n**Detailed Solution:**\n{csv_ans_clean}"
#         else:
#             ref_ans = hf_ans or csv_ans_clean
#     else:
#         question_text = flatten_latex_for_argilla(csv_question)
#         context_input = flatten_latex_for_argilla(csv_context)
#         images_rendered = ""
#         image_desc = ""
#         ref_ans = flatten_latex_for_argilla(csv_ans)
        
#     model_ans = flatten_latex_for_argilla(model_ans)

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
import re
import json
import csv
import zipfile
import tempfile
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from inspect_ai.log import read_eval_log

def is_in_our_subsets(sample_id, metadata):
    sid = str(sample_id).lower()
    meta_school = str(metadata.get('school_type', '')).lower()
    if 'gel' not in sid and meta_school != 'gel': return False
    year = str(metadata.get('year', ''))
    if not (any(y in sid for y in ['2020', '2021']) or year in ['2020', '2021']): return False
    
    meta_subject = str(metadata.get('subject', '')).lower()
    is_chemistry = 'chemistry' in sid or 'chemistry' in meta_subject or 'ximeia' in sid or 'ximeia' in meta_subject
    is_biology = 'biology' in sid or 'biology' in meta_subject or 'biologia' in sid or 'biologia' in meta_subject
    return is_chemistry or is_biology

def get_subject_name(sample_id, metadata):
    combined = (str(sample_id) + " " + str(metadata.get('subject', ''))).lower()
    if 'chemistry' in combined or 'ximeia' in combined: return 'Chemistry'
    elif 'biology' in combined or 'biologia' in combined: return 'Biology'
    return 'Other'

# Ο ΜΙΝΙΜΑΛ ΚΑΙ ΣΤΟΧΕΥΜΕΝΟΣ ΚΑΘΑΡΙΣΜΟΣ ΓΙΑ ΤΕΛΕΙΟ KaTeX RENDERING
def clean_science_latex(text):
    if not text or pd.isna(text): return ""
    text = str(text)

    # 1. Καθαρισμός tags των συλλογιστικών μοντέλων
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 2. Χειρισμός παρενθέσεων μοντέλου
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    text = text.replace(r'\[', '$$').replace(r'\]', '$$')

    # 3. Διορθώσεις ΣΤΟΧΕΥΜΕΝΩΝ λαθών που προκαλούν Red KaTeX ParseError
    bugs = {
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
        r"\mathrm{H_{3}$O+}": r"H_{3}O^{+}",
        r"\mathrm{H_{3}$O+}": r"H_{3}O^{+}",
        r"H_{3}$O+": r"H_{3}O^{+}",
        r"\mathrm{H\Delta}": r"H\Delta",
        r"\mathrm{\Delta^-}": r"\Delta^-",
        r"NO_{2}$ -": r"NO_{2}^{-}",
        r"C_{6}H_{5}-": r"C_{6}H_{5}^{-}",
        r"HO-": r"HO^{-}",
        r"λόγος \frac{[CH_3COO^-]}{[CH_3COOH]} = 1 $": r"λόγος $\frac{[CH_3COO^-]}{[CH_3COOH]} = 1$",
    }
    for bad, good in bugs.items():
        text = text.replace(bad, good)

    # 4. Μετάφραση Χημείας (\ce -> \mathrm)
    # Το Argilla/KaTeX δεν υποστηρίζει εγγενώς το mhchem, οπότε το φτιάχνουμε εμείς
    def chem_replacer(match):
        formula = match.group(1).replace('$', '') # Αφαιρεί τα $ μέσα στη χημεία που κρασάρουν το σύστημα
        formula = re.sub(r'(?<=[A-Za-z)\]])(\d+)', r'_{\1}', formula) # Αριθμοί σε δείκτες
        formula = re.sub(r'([+-]+)$', r'^{\1}', formula) # Φορτία σε εκθέτες
        return f"\\mathrm{{{formula}}}"
    
    text = re.sub(r'\\ce\s*{([^}]+)}', chem_replacer, text)
    text = re.sub(r'\\ce\s*([A-Za-z0-9_+\-\^]+)', chem_replacer, text)

    # 5. Βελάκια και πολλαπλασιασμός
    text = text.replace('< = >', r' \rightleftharpoons ').replace('<=>', r' \rightleftharpoons ')
    text = text.replace(' - > ', r' \rightarrow ').replace(' -> ', r' \rightarrow ')
    text = text.replace(r'\cdotpmin', r'\cdot \mathrm{min}').replace(r'\cdotp', r'\cdot')

    # 6. Καθαρισμός ορφανών $ στο τέλος
    text = re.sub(r'(?<!\$)\$\s*$', '', text)
    
    # 7. Διατήρηση όμορφων παραγράφων (χωρίς να πειράζουμε τα $$)
    text = text.replace('\r', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def clean_prefix(text):
    text = clean_science_latex(text)
    return re.sub(r'^(Question|Context|Answer|Ερώτηση|Απάντηση|Κείμενο):\s*', '', text, flags=re.IGNORECASE).strip()

def parse_input_to_dict(text, img_desc):
    text = clean_science_latex(text)
    parsed = {}
    text = re.sub(r'\[Image Description:.*?\]', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'\[Image Transcription:.*?\]', '', text, flags=re.IGNORECASE).strip()
    
    has_context = "Context:" in text
    has_question = "Question:" in text
    
    if has_context and has_question:
        parts = text.split("Question:")
        parsed["context"] = parts[0].replace("Context:", "").strip()
        parsed["question"] = parts[1].strip()
    elif has_question:
        parsed["question"] = text.replace("Question:", "").strip()
    elif has_context:
        parsed["context"] = text.replace("Context:", "").strip()
    else:
        parsed["question"] = text.strip()
        
    if img_desc: parsed["image_description"] = clean_science_latex(img_desc)
    return parsed

def build_dataframe(log_dir):
    log_dir_path = Path(log_dir)
    rows = []
    
    target_models = ['krikri', 'llama', 'gemma', 'qwen']
    model_names_mapping = {
        'krikri': 'krikri-8b-instruct-zero-shot',
        'llama': 'llama-3.1-8b-instruct-zero-shot',
        'gemma': 'gemma-4-26b-it-zero-shot',
        'qwen': 'qwen3-32b-zero-shot'
    }

    zip_files = list(log_dir_path.rglob('*.zip'))
    for zip_path in zip_files:
        fn_lower = zip_path.name.lower()
        matched_model = next((m for m in target_models if m in fn_lower), None)
        model_name = model_names_mapping[matched_model] if matched_model else zip_path.stem
            
        print(f"\nProcessing archive: {zip_path.name} (Mapped Model: {model_name})")
        
        with zipfile.ZipFile(zip_path) as outer_zf:
            eval_entries = [n for n in outer_zf.namelist() if n.endswith('.eval') and 'panellinies' in n.lower() and '0-shot' in n.lower()]
            if not eval_entries:
                eval_entries = [n for n in outer_zf.namelist() if n.endswith('.eval') and '0-shot' in n.lower()]

            for eval_entry in eval_entries:
                print(f"  Reading eval: {Path(eval_entry).name}")
                eval_bytes = outer_zf.read(eval_entry)
                
                with tempfile.NamedTemporaryFile(suffix='.eval', delete=True) as tmp:
                    tmp.write(eval_bytes)
                    tmp.flush()
                    try: eval_log = read_eval_log(tmp.name)
                    except Exception as e: continue
                    
                    for sample in eval_log.samples:
                        sample_id = sample.id
                        metadata = sample.metadata or {}
                        if metadata.get('format') != 'open_ended': continue
                        
                        if is_in_our_subsets(sample_id, metadata):
                            subject = get_subject_name(sample_id, metadata)
                            scores = sample.scores or {}
                            
                            raw_input = str(sample.input)
                            raw_target = str(sample.target)
                            raw_answer = str(sample.output.completion) if sample.output else ""
                            img_desc = metadata.get('image_description')
                            
                            input_dict = parse_input_to_dict(raw_input, img_desc)
                            target_dict = {"reference": clean_prefix(raw_target)}
                            answer_dict = {"answer": clean_prefix(raw_answer)}
                            
                            judge_score = getattr(scores.get('generic_judge_scorer', object()), 'value', None)
                            bert_score = getattr(scores.get('greek_bertscore', object()), 'value', None)

                            rows.append({
                                'Subject': subject,
                                'Model': model_name,
                                'Question_ID': sample_id,
                                'Input_Question': json.dumps(input_dict, ensure_ascii=False),
                                'Reference_Target': json.dumps(target_dict, ensure_ascii=False),
                                'Model_Answer': json.dumps(answer_dict, ensure_ascii=False),
                                'LLM_Judge_Score': judge_score,
                                'BERTScore': bert_score
                            })
                            
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset=['Model', 'Question_ID'], keep='last')
    return df

if __name__ == '__main__':
    load_dotenv()
    logs_dir = os.getenv('PANELLINIES_LOGS_DIR', '/home/eleni/panellinies/logs')
    
    print(f"Starting parsing in: {logs_dir}")
    final_df = build_dataframe(logs_dir)
    
    print(f"\nTotal records collected: {len(final_df)}")
    if not final_df.empty:
        output_file = 'human_evaluation_panellinies_science_subset.csv'
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        print(f"File created successfully: {output_file}")
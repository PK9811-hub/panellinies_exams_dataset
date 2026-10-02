# import os
# import io
# import re
# import json
# import csv
# import zipfile
# import tempfile
# from pathlib import Path
# import pandas as pd
# from dotenv import load_dotenv
# from inspect_ai.log import read_eval_log

# def is_in_our_subsets(sample_id, metadata):
#     sid = str(sample_id).lower()
#     meta_school = str(metadata.get('school_type', '')).lower()
    
#     # 1. School Type: GEL
#     if 'gel' not in sid and meta_school != 'gel':
#         return False

#     # 2. Χρονιές: 2021 ΚΑΙ 2022
#     year = str(metadata.get('year', ''))
#     if not (any(y in sid for y in ['2021', '2022']) or year in ['2021', '2022']):
#         return False

#     # 3. Μαθήματα
#     meta_subject = str(metadata.get('subject', '')).lower()
#     is_ancient = 'ancient_greek' in sid or 'ancient_greek' in meta_subject
#     is_latin = 'latin' in sid or 'latin' in meta_subject
#     is_modern = 'greek_language' in sid or 'greek_language' in meta_subject

#     return is_ancient or is_latin or is_modern

# def get_subject_name(sample_id, metadata):
#     combined = (str(sample_id) + " " + str(metadata.get('subject', ''))).lower()
#     if 'ancient_greek' in combined:
#         return 'Ancient Greek'
#     elif 'latin' in combined:
#         return 'Latin'
#     elif 'greek_language' in combined:
#         return 'Modern Greek'
#     return 'Other'

# def sanitize_text(text):
#     if not isinstance(text, str):
#         text = str(text) if text is not None else ""
        
#     # 1. Αφαίρεση του Chain of Thought (Qwen / Deepseek)
#     # Αφαιρεί τα ολοκληρωμένα <think> ... </think>
#     text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
#     # Αφαιρεί το <think> αν δεν έκλεισε ποτέ λόγω κοψίματος (max tokens)
#     text = re.sub(r'<think>.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    
#     # 2. Προστασία του Excel/Argilla από σπάσιμο γραμμών
#     # Αντικαθιστούμε κάθε είδος αλλαγής γραμμής με ένα κενό διάστημα
#     text = text.replace('\r', ' ')
#     text = text.replace('\n', ' ')
#     text = text.replace('\\n', ' ') 
    
#     # Καθαρισμός πολλαπλών συνεχόμενων κενών που ίσως δημιουργήθηκαν
#     text = re.sub(r'\s+', ' ', text)
    
#     return text.strip()

# def clean_prefix(text):
#     text = sanitize_text(text)
#     return re.sub(r'^(Question|Context|Answer):\s*', '', text, flags=re.IGNORECASE).strip()

# def parse_input_to_dict(text, img_desc):
#     text = sanitize_text(text)
#     parsed = {}
    
#     # Καθαρισμός image tags
#     text = re.sub(r'\[Image Description:.*?\]', '', text, flags=re.IGNORECASE).strip()
#     text = re.sub(r'\[Image Transcription:.*?\]', '', text, flags=re.IGNORECASE).strip()
    
#     has_context = "Context:" in text
#     has_question = "Question:" in text
    
#     if has_context and has_question:
#         parts = text.split("Question:")
#         parsed["context"] = parts[0].replace("Context:", "").strip()
#         parsed["question"] = parts[1].strip()
#     elif has_question:
#         parsed["question"] = text.replace("Question:", "").strip()
#     elif has_context:
#         parsed["context"] = text.replace("Context:", "").strip()
#     else:
#         parsed["question"] = text.strip()
        
#     if img_desc:
#         parsed["image_description"] = sanitize_text(img_desc).strip()
        
#     return parsed

# def build_dataframe(log_dir):
#     log_dir_path = Path(log_dir)
#     rows = []
    
#     # Αυστηρό Mapping Ονομάτων
#     target_models = ['krikri', 'llama', 'gemma', 'qwen']
#     model_names_mapping = {
#         'krikri': 'krikri-8b-instruct-zero-shot',
#         'llama': 'llama-3.1-8b-instruct-zero-shot',
#         'gemma': 'gemma-4-26b-it-zero-shot',
#         'qwen': 'qwen3-32b-zero-shot'
#     }

#     zip_files = list(log_dir_path.rglob('*.zip'))
    
#     for zip_path in zip_files:
#         fn_lower = zip_path.name.lower()
        
#         # Εντοπισμός μοντέλου και απόδοση του αυστηρού ονόματος
#         matched_model = next((m for m in target_models if m in fn_lower), None)
#         if matched_model:
#             model_name = model_names_mapping[matched_model]
#         else:
#             model_name = zip_path.stem # Fallback
            
#         print(f"\nProcessing archive: {zip_path.name} (Mapped Model: {model_name})")
        
#         with zipfile.ZipFile(zip_path) as outer_zf:
#             eval_entries = [
#                 n for n in outer_zf.namelist() 
#                 if n.endswith('.eval') and 'panellinies' in n.lower() and '0-shot' in n.lower()
#             ]
            
#             if not eval_entries:
#                 eval_entries = [n for n in outer_zf.namelist() if n.endswith('.eval') and '0-shot' in n.lower()]

#             for eval_entry in eval_entries:
#                 print(f"  Reading eval: {Path(eval_entry).name}")
#                 eval_bytes = outer_zf.read(eval_entry)
                
#                 with tempfile.NamedTemporaryFile(suffix='.eval', delete=True) as tmp:
#                     tmp.write(eval_bytes)
#                     tmp.flush()
                    
#                     try:
#                         eval_log = read_eval_log(tmp.name)
#                     except Exception as e:
#                         print(f"  [Error reading {eval_entry}]: {e}")
#                         continue
                    
#                     for sample in eval_log.samples:
#                         sample_id = sample.id
#                         metadata = sample.metadata or {}
                        
#                         if metadata.get('format') != 'open_ended':
#                             continue
                        
#                         if is_in_our_subsets(sample_id, metadata):
#                             subject = get_subject_name(sample_id, metadata)
#                             scores = sample.scores or {}
                            
#                             raw_input = sample.input if isinstance(sample.input, str) else str(sample.input)
#                             raw_target = sample.target if isinstance(sample.target, str) else str(sample.target)
                            
#                             raw_answer = ""
#                             if sample.output:
#                                 raw_answer = getattr(sample.output, 'completion', '') or str(sample.output)
                                
#                             img_desc = metadata.get('image_description')
                            
#                             input_dict = parse_input_to_dict(raw_input, img_desc)
#                             target_dict = {"reference": clean_prefix(raw_target)}
#                             answer_dict = {"answer": clean_prefix(raw_answer)}
                            
#                             judge_score = None
#                             bert_score = None
#                             if 'generic_judge_scorer' in scores:
#                                 judge_score = getattr(scores['generic_judge_scorer'], 'value', None)
#                             if 'greek_bertscore' in scores:
#                                 bert_score = getattr(scores['greek_bertscore'], 'value', None)

#                             rows.append({
#                                 'Subject': subject,
#                                 'Model': model_name,
#                                 'Question_ID': sample_id,
#                                 'Input_Question': json.dumps(input_dict, ensure_ascii=False),
#                                 'Reference_Target': json.dumps(target_dict, ensure_ascii=False),
#                                 'Model_Answer': json.dumps(answer_dict, ensure_ascii=False),
#                                 'LLM_Judge_Score': judge_score,
#                                 'BERTScore': bert_score
#                             })
                            
#     df = pd.DataFrame(rows)
#     if not df.empty:
#         df = df.drop_duplicates(subset=['Model', 'Question_ID'], keep='last')
        
#     return df

# if __name__ == '__main__':
#     load_dotenv()
#     logs_dir = os.getenv('PANELLINIES_LOGS_DIR', '/home/eleni/panellinies/logs')
    
#     print(f"Starting parsing in: {logs_dir}")
#     final_df = build_dataframe(logs_dir)
    
#     print(f"\nTotal records collected (2021 & 2022): {len(final_df)}")
#     if not final_df.empty:
#         # Το νέο όνομα του αρχείου που ζητήθηκε
#         output_file = 'human_evaluation_panellinies_hum_subset.csv'
#         final_df.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
#         print(f"File created successfully: {output_file}")
#         print("\nBreakdown by Subject and Model:")
#         print(final_df.groupby(['Subject', 'Model']).size())





# import os
# import io
# import re
# import json
# import csv
# import zipfile
# import tempfile
# from pathlib import Path
# import pandas as pd
# from dotenv import load_dotenv
# from inspect_ai.log import read_eval_log

# def is_in_our_subsets(sample_id, metadata):
#     sid = str(sample_id).lower()
#     meta_school = str(metadata.get('school_type', '')).lower()
    
#     # 1. School Type: GEL
#     if 'gel' not in sid and meta_school != 'gel':
#         return False

#     # 2. Χρονιές: 2021 ΚΑΙ 2022
#     year = str(metadata.get('year', ''))
#     if not (any(y in sid for y in ['2021', '2022']) or year in ['2021', '2022']):
#         return False

#     # 3. Μαθήματα
#     meta_subject = str(metadata.get('subject', '')).lower()
#     is_ancient = 'ancient_greek' in sid or 'ancient_greek' in meta_subject
#     is_latin = 'latin' in sid or 'latin' in meta_subject
#     is_modern = 'greek_language' in sid or 'greek_language' in meta_subject

#     return is_ancient or is_latin or is_modern

# def get_subject_name(sample_id, metadata):
#     combined = (str(sample_id) + " " + str(metadata.get('subject', ''))).lower()
#     if 'ancient_greek' in combined:
#         return 'Ancient Greek'
#     elif 'latin' in combined:
#         return 'Latin'
#     elif 'greek_language' in combined:
#         return 'Modern Greek'
#     return 'Other'

# def sanitize_text(text, max_len=25000):
#     if not isinstance(text, str):
#         text = str(text) if text is not None else ""
        
#     # 1. Καθαρισμός τυχόν tags <think>
#     text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
#     text = re.sub(r'<think>.*', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    
#     # 2. Αφαίρεση του Chain of Thought (οι αγγλικές "ασυναρτησίες" του Qwen)
#     # Χωρίζουμε το κείμενο με βάση τα newlines (πραγματικά ή escaped)
#     chunks = re.split(r'(?:\\n|\n|\r)+', text)
#     valid_chunks = []
    
#     # Λέξεις κλειδιά που χρησιμοποιεί το Qwen όταν μιλάει στον εαυτό του
#     cot_markers = ['hmm,', 'wait,', 'let me', 'perhaps the', 'alternatively,', 'so the ', 'this means', 'i need to', 'let\'s ']
    
#     for chunk in chunks:
#         c_lower = chunk.strip().lower()
#         if not c_lower:
#             continue
            
#         # Αν η πρόταση ξεκινάει ή περιέχει κλασικές εκφράσεις CoT του Qwen, αγνόησέ την
#         if any(c_lower.startswith(m) for m in cot_markers) or any(f" {m}" in c_lower for m in cot_markers):
#             continue
            
#         # Αν είναι μεγάλο κομμάτι κειμένου ΠΛΗΡΩΣ στα αγγλικά (χωρίς ΚΑΝΕΝΑ ελληνικό γράμμα)
#         has_greek = bool(re.search(r'[α-ωΑ-ΩάέήίόύώΆΈΉΊΌΎΏ]', chunk))
#         if not has_greek and len(chunk) > 100:
#             continue
            
#         valid_chunks.append(chunk.strip())
        
#     text = " ".join(valid_chunks)
    
#     # 3. Εξαφάνιση όλων των αλλαγών γραμμής (πραγματικών & escaped) για προστασία του Excel
#     text = text.replace('\n', ' ').replace('\r', ' ')
#     text = text.replace('\\n', ' ').replace('\\r', ' ')
#     text = text.replace('\\', '') # Αφαιρούμε τα backslashes για να μη χαλάσουν το JSON format
    
#     # 4. Συμπύκνωση πολλαπλών κενών
#     text = re.sub(r'\s+', ' ', text)
    
#     # 5. ΑΥΣΤΗΡΟ ΟΡΙΟ (Προστασία από το όριο των 32.767 χαρακτήρων του Excel)
#     if len(text) > max_len:
#         text = text[:max_len] + "... [TRUNCATED DUE TO LENGTH]"
        
#     return text.strip()

# def clean_prefix(text):
#     text = sanitize_text(text)
#     return re.sub(r'^(Question|Context|Answer):\s*', '', text, flags=re.IGNORECASE).strip()

# def parse_input_to_dict(text, img_desc):
#     text = sanitize_text(text)
#     parsed = {}
    
#     text = re.sub(r'\[Image Description:.*?\]', '', text, flags=re.IGNORECASE).strip()
#     text = re.sub(r'\[Image Transcription:.*?\]', '', text, flags=re.IGNORECASE).strip()
    
#     has_context = "Context:" in text
#     has_question = "Question:" in text
    
#     if has_context and has_question:
#         parts = text.split("Question:")
#         parsed["context"] = parts[0].replace("Context:", "").strip()
#         parsed["question"] = parts[1].strip()
#     elif has_question:
#         parsed["question"] = text.replace("Question:", "").strip()
#     elif has_context:
#         parsed["context"] = text.replace("Context:", "").strip()
#     else:
#         parsed["question"] = text.strip()
        
#     if img_desc:
#         parsed["image_description"] = sanitize_text(img_desc).strip()
        
#     return parsed

# def build_dataframe(log_dir):
#     log_dir_path = Path(log_dir)
#     rows = []
    
#     # Αυστηρό mapping ονομάτων για να μην κρασάρει το Argilla
#     target_models = ['krikri', 'llama', 'gemma', 'qwen']
#     model_names_mapping = {
#         'krikri': 'krikri-8b-instruct-zero-shot',
#         'llama': 'llama-3.1-8b-instruct-zero-shot',
#         'gemma': 'gemma-4-26b-it-zero-shot',
#         'qwen': 'qwen3-32b-zero-shot'
#     }

#     zip_files = list(log_dir_path.rglob('*.zip'))
    
#     for zip_path in zip_files:
#         fn_lower = zip_path.name.lower()
        
#         matched_model = next((m for m in target_models if m in fn_lower), None)
#         if matched_model:
#             model_name = model_names_mapping[matched_model]
#         else:
#             model_name = zip_path.stem
            
#         print(f"\nProcessing archive: {zip_path.name} (Mapped Model: {model_name})")
        
#         with zipfile.ZipFile(zip_path) as outer_zf:
#             eval_entries = [
#                 n for n in outer_zf.namelist() 
#                 if n.endswith('.eval') and 'panellinies' in n.lower() and '0-shot' in n.lower()
#             ]
            
#             if not eval_entries:
#                 eval_entries = [n for n in outer_zf.namelist() if n.endswith('.eval') and '0-shot' in n.lower()]

#             for eval_entry in eval_entries:
#                 print(f"  Reading eval: {Path(eval_entry).name}")
#                 eval_bytes = outer_zf.read(eval_entry)
                
#                 with tempfile.NamedTemporaryFile(suffix='.eval', delete=True) as tmp:
#                     tmp.write(eval_bytes)
#                     tmp.flush()
                    
#                     try:
#                         eval_log = read_eval_log(tmp.name)
#                     except Exception as e:
#                         print(f"  [Error reading {eval_entry}]: {e}")
#                         continue
                    
#                     for sample in eval_log.samples:
#                         sample_id = sample.id
#                         metadata = sample.metadata or {}
                        
#                         if metadata.get('format') != 'open_ended':
#                             continue
                        
#                         if is_in_our_subsets(sample_id, metadata):
#                             subject = get_subject_name(sample_id, metadata)
#                             scores = sample.scores or {}
                            
#                             raw_input = sample.input if isinstance(sample.input, str) else str(sample.input)
#                             raw_target = sample.target if isinstance(sample.target, str) else str(sample.target)
                            
#                             raw_answer = ""
#                             if sample.output:
#                                 raw_answer = getattr(sample.output, 'completion', '') or str(sample.output)
                                
#                             img_desc = metadata.get('image_description')
                            
#                             # Δημιουργία JSON objects
#                             input_dict = parse_input_to_dict(raw_input, img_desc)
#                             target_dict = {"reference": clean_prefix(raw_target)}
#                             answer_dict = {"answer": clean_prefix(raw_answer)}
                            
#                             judge_score = None
#                             bert_score = None
#                             if 'generic_judge_scorer' in scores:
#                                 judge_score = getattr(scores['generic_judge_scorer'], 'value', None)
#                             if 'greek_bertscore' in scores:
#                                 bert_score = getattr(scores['greek_bertscore'], 'value', None)

#                             rows.append({
#                                 'Subject': subject,
#                                 'Model': model_name,
#                                 'Question_ID': sample_id,
#                                 'Input_Question': json.dumps(input_dict, ensure_ascii=False),
#                                 'Reference_Target': json.dumps(target_dict, ensure_ascii=False),
#                                 'Model_Answer': json.dumps(answer_dict, ensure_ascii=False),
#                                 'LLM_Judge_Score': judge_score,
#                                 'BERTScore': bert_score
#                             })
                            
#     df = pd.DataFrame(rows)
#     if not df.empty:
#         df = df.drop_duplicates(subset=['Model', 'Question_ID'], keep='last')
        
#     return df

# if __name__ == '__main__':
#     load_dotenv()
#     logs_dir = os.getenv('PANELLINIES_LOGS_DIR', '/home/eleni/panellinies/logs')
    
#     print(f"Starting parsing in: {logs_dir}")
#     final_df = build_dataframe(logs_dir)
    
#     print(f"\nTotal records collected (2021 & 2022): {len(final_df)}")
#     if not final_df.empty:
#         output_file = 'human_evaluation_panellinies_hum_subset.csv'
#         final_df.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
#         print(f"File created successfully: {output_file}")
#         print("\nBreakdown by Subject and Model:")
#         print(final_df.groupby(['Subject', 'Model']).size())


import os
import io
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

def flatten_latex(text):
    if not isinstance(text, str): return ""
    
    # 1. Καθαρισμός <think>
    text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<think>.*', ' ', text, flags=re.IGNORECASE | re.DOTALL)
    
    # 2. ΑΠΕΝΕΡΓΟΠΟΙΗΣΗ ΤΟΥ KATEX: Σβήνουμε όλα τα σύμβολα που ανοίγουν μαθηματικά
    text = text.replace('$', '')
    text = text.replace(r'\(', '').replace(r'\)', '')
    text = text.replace(r'\[', '').replace(r'\]', '')
    
    # 3. Αφαίρεση τοξικών εντολών LaTeX (κρατάμε μόνο το καθαρό περιεχόμενό τους)
    for _ in range(3): # Επανάληψη για nested εντολές π.χ. \mathrm{\ce{...}}
        text = re.sub(r'\\ce{([^}]+)}', r'\1', text)
        text = re.sub(r'\\mathrm{([^}]+)}', r'\1', text)
        text = re.sub(r'\\text{([^}]+)}', r'\1', text)
        text = re.sub(r'\\mathbf{([^}]+)}', r'\1', text)
    
    # Αφαίρεση \ce χωρίς αγκύλες (π.χ. \ceH2O -> H2O)
    text = re.sub(r'\\ce([A-Za-z0-9_+\-]+)', r'\1', text)
    
    # 4. Μετατροπή βελών και συμβόλων σε απλό text
    text = text.replace(r'\rightleftharpoons', '<=>')
    text = text.replace(r'\rightarrow', '->')
    text = text.replace(r'\Rightarrow', '=>')
    text = text.replace(r'\cdot', '*')
    text = text.replace('< = >', '<=>')
    text = text.replace(' - > ', '->')
    
    # 5. Καθαρισμός κενών και escape characters
    text = text.replace(r'\,', ' ').replace(r'\;', ' ').replace(r'\ ', ' ')
    text = text.replace(r'\%', '%').replace(r'\_', '_')
    
    return text.strip()

def sanitize_text(text):
    text = flatten_latex(text)
    
    # Αφαίρεση του Chain of Thought (Qwen)
    chunks = re.split(r'(?:\\n|\n|\r)+', text)
    valid_chunks = []
    cot_markers = ['hmm,', 'wait,', 'let me', 'perhaps the', 'alternatively,', 'so the ', 'this means', 'i need to', 'let\'s ']
    
    for chunk in chunks:
        c_lower = chunk.strip().lower()
        if not c_lower: continue
        if any(c_lower.startswith(m) for m in cot_markers) or any(f" {m}" in c_lower for m in cot_markers):
            continue
        valid_chunks.append(chunk.strip())
        
    text = " ".join(valid_chunks)
    
    # Προστασία του Excel/Argilla από σπάσιμο γραμμών
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def clean_prefix(text):
    text = sanitize_text(text)
    return re.sub(r'^(Question|Context|Answer):\s*', '', text, flags=re.IGNORECASE).strip()

def parse_input_to_dict(text, img_desc):
    text = sanitize_text(text)
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
        
    if img_desc:
        parsed["image_description"] = sanitize_text(img_desc).strip()
        
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
                    
                    try:
                        eval_log = read_eval_log(tmp.name)
                    except Exception as e:
                        print(f"  [Error reading {eval_entry}]: {e}")
                        continue
                    
                    for sample in eval_log.samples:
                        sample_id = sample.id
                        metadata = sample.metadata or {}
                        
                        if metadata.get('format') != 'open_ended': continue
                        
                        if is_in_our_subsets(sample_id, metadata):
                            subject = get_subject_name(sample_id, metadata)
                            scores = sample.scores or {}
                            
                            raw_input = sample.input if isinstance(sample.input, str) else str(sample.input)
                            raw_target = sample.target if isinstance(sample.target, str) else str(sample.target)
                            raw_answer = getattr(sample.output, 'completion', '') or str(sample.output) if sample.output else ""
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
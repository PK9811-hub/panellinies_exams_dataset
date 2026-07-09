import os
import json
import re
import pandas as pd
import argparse
from pathlib import Path
import ast
import random
import argparse
import logging
import warnings
warnings.filterwarnings('ignore', category=SyntaxWarning, message='invalid escape sequence')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
from datasets import Dataset, Image, Features, Sequence 
from huggingface_hub import create_repo, repo_exists, HfApi 
from dotenv import load_dotenv, find_dotenv 

load_dotenv(find_dotenv())

# --- HELPER FUNCTIONS ---

# --- DATA LOADERS ---
#identification of year, school_type, and subject based on filepath
def parse_path_metadata(filepath):
    path_obj = Path(filepath)
    
    try:
        data_index = path_obj.parts.index('data')
        year = path_obj.parts[data_index + 1]        
        school_type = path_obj.parts[data_index + 2]  
        subject_folder = path_obj.parts[data_index + 3] 
        
        return subject_folder, school_type, year
    except (ValueError, IndexError):
        return None, None, None

#parsing json files' structure
def parse_json_questions(json_path):
    questions_list = []
    json_path = Path(json_path)
    
    if not json_path.exists():
        logger.warning(f'Προσοχή: Το αρχείο {json_path} δε βρέθηκε.')
        return questions_list
    
    with json_path.open ("r", encoding = "utf-8-sig") as f:
        data = json.load(f)
        return data

#parsing md files' structure and creation of dict with answers: {"ID": "Απάντηση"}
def parse_markdown_answers(md_path):
    answers_dict = {}
    md_path = Path(md_path)
    
    if not md_path.exists():
        logger.warning(f"Προσοχή: Το αρχείο {md_path} δε βρέθηκε.")
        return answers_dict
        
    with md_path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
                
            parts = line.split("\t", 1)
            
            if len(parts) == 1:
                parts = line.split(maxsplit=1)
                
            if len(parts) == 2:
                q_id = parts[0].strip()
                ans_text = parts[1].strip()
                
                if ans_text.startswith('"') and ans_text.endswith('"'):
                    try:
                        ans_text = ast.literal_eval(ans_text)
                    except (SyntaxError, ValueError):
                        ans_text = ans_text[1:-1].strip()
                        ans_text = ans_text.replace('\\n', '\n')
                
                answers_dict[q_id] = ans_text
                
    return answers_dict

#merging questions (json) with answers (md) based on ids and adding metadata
def merge_qa_data(json_path, md_path):
    combined_data = []
    
    subject, school_type, year = parse_path_metadata(json_path)
    questions_list = parse_json_questions(json_path)
    answers_dict = parse_markdown_answers(md_path)
    
    for question in questions_list:
        current_q_id = question["id"]
        
        if current_q_id in answers_dict:
            question["answer"] = answers_dict[current_q_id]
        else:
            question["answer"] = None
            logger.warning(f"Λείπει η απάντηση για το {current_q_id} στο μάθημα {subject} ({year})")
        
        question["subject"] = subject
        question["year"] = year
        question["school_type"] = school_type
        
        combined_data.append(question)
    
    return combined_data

#folder_scanning and file_pairing returning a list of questions with their answers
def get_file_pairs(data_dir, target_school=None):
    data_dir = Path(data_dir)
    pairs = []
    
    for json_path in data_dir.rglob("*.json"):
        if target_school and target_school not in json_path.parts:
            continue
        
        json_filename = json_path.name
        md_filename = json_filename.replace("them_", "apant_").replace(".json", ".md")
        
        md_path = json_path.parent / md_filename
        
        if md_path.exists():
            pairs.append({"json": json_path, "md": md_path})
        else:
            logger.warning(f"Βρέθηκε το {json_filename} χωρίς απάντηση.")
            
    return pairs

# --- DATA TRANSFORMERS ---        

def detect_exercise_type(q_text, q_choices):
    if q_text:
        question_text = str(q_text).lower() 
    else:
        question_text = ""
        
    if q_choices:
        choices_list = list(q_choices)
    else:
        choices_list = []
    
    # 1. fill_in_the_gaps
    if "κενά" in question_text or "...." in question_text:
        return "fill_in_the_gaps"
    
    # 2. matching 
    has_keyword = False
    if "στήλη" in question_text or "στήλης" in question_text:
        has_keyword = True
    
    has_numbers = bool(re.search(r'\d+\.', question_text))
    has_letters = bool(re.search(r'[α-ω]\.', question_text))
    
    if has_keyword and has_numbers and has_letters:
        return "matching"
    
    # 3. open_ended
    if len(choices_list) == 0:
        return "open_ended"
    
    # 4. true_false
    if len(choices_list) <= 2:
        for choice in choices_list:
            current_choice = str(choice).lower()
            true_false_keywords = "σωστό" in current_choice or "λάθος" in current_choice or "αληθής" in current_choice or "ψευδής" in current_choice
        
            if true_false_keywords:
                return "true_false"
    
    # 5. multiple_choice
    return "multiple_choice"

def unify_label(answer_text, form_type):
    if form_type != 'multiple_choice' and form_type != "true_false":
        return answer_text
    
    clean_ans = str(answer_text).replace(".","").replace(")","").replace("(","").strip().lower()
    mapping = {"α": "A", "β": "B", "γ": "C", "δ": "D", "ε": "E", "i": "A", "ii": "B", "iii": "C", "iv": "D", "σωστό": "True", "λάθος": "False"}
    
    return mapping.get(clean_ans, clean_ans)

def find_answer_index(choices, answer_text):
    if choices == [] or answer_text == "":
        return None
    
    answer_str = str(answer_text).strip().lower()
    
    for idx, choice in enumerate(choices):
        choice_str = str(choice).strip().lower()
        
        if choice_str == answer_str:
            return idx
            
        if choice_str.startswith(f"{answer_str}.") or choice_str.startswith(f"{answer_str})") or choice_str.startswith(f"{answer_str} "):
            return idx
            
        if choice_str.endswith(answer_str):
            return idx
            
    return None     

def apply_reference_tag(item):
    image_urls = item.get("images", [])
    input_text = str(item.get("input", "")).lower()
    subj = str(item.get("subject", "")).lower()
    
    if image_urls != []:
        return "multimodal"
    
    elif "πίνακα" in input_text or "πίνακας" in input_text:
        return "table"
    
    elif len(input_text) > 200 and subj in ["arxaia", "istoria", "latinika", "nea_ellinika"]:
        return "passage"
    
    else:
        return "none"    

def normalize_for_compare(value):
    if isinstance(value, str) and value.startswith("["):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
    if isinstance (value, (list,tuple)):
        return str([str(x).strip() for x in value])
    try:
        if pd.isna(value):
            return ""
    except ValueError:
        pass
    return str(value).strip()

# --- DATA CONSOLIDATION ---
def consolidate(target_school="GEL", output_filename="panellinies_dataset.xlsx"):
    """
    Reads, cleans, transforms and exports the whole dataset in an Excel file.
    """
    data_path = os.getenv("DATA_DIR")
    if not data_path:
        logger.error("Το DATA_DIR δεν βρέθηκε στο .env αρχείο!")
        return pd.DataFrame() 

    data_dir = Path(data_path)
    logger.info(f"🚀 Ξεκινάει η αναζήτηση στον φάκελο: {data_dir}")
    
    all_pairs = get_file_pairs(data_dir, target_school=target_school)
    logger.info(f"Βρέθηκαν συνολικά {len(all_pairs)} ζευγάρια αρχείων (JSON/MD).")

    main_dataset = []
    for pair in all_pairs:
        json_path = pair["json"]
        md_path = pair["md"]
        qa_list = merge_qa_data(json_path, md_path)
        main_dataset.extend(qa_list)

    logger.info(f"Η ενοποίηση ολοκληρώθηκε! Βρέθηκαν συνολικά {len(main_dataset)} ερωτήσεις-απαντήσεις.")

    subject_translation = {
        "nea_ellinika": "greek_language",
        "arxaia": "ancient_greek",
        "istoria": "history",
        "latinika": "latin",
        "biologia": "biology",
        "fysiki": "physics",
        "ximeia": "chemistry",
        "pliroforiki": "computer_science",
        "arxes_oikonomikis_theorias": "economics",
        "mathimatika": "mathematics"
    }

    images_found = 0 

    for item in main_dataset:
        q_text = item.get("question", "")
        q_choices = item.get("choices", [])
        ans_text = item.get("answer", "")
        images_list = item.get("images", [])
        marks = item.get("mark", [])
        
        # format & answers & reference
        item["format"] = detect_exercise_type(q_text, q_choices)
        item["answer_index"] = find_answer_index(q_choices, ans_text)
        item["reference"] = apply_reference_tag(item)
        
        # subject translation
        old_subj = item.get("subject", "")
        new_subj = subject_translation.get(old_subj, old_subj)
        item["subject"] = new_subj
        
        # image processing
        all_descriptions = []
        all_transcriptions = []
        all_paths = []
        
        for img_dict in images_list:
            desc = img_dict.get("description", "")
            if desc:
                all_descriptions.append(desc)
                
            transc = img_dict.get("transcription", [])
            if transc and isinstance(transc, list):
                joined_transc = ", ".join(transc)
                all_transcriptions.append(joined_transc)
            
            img_path = img_dict.get("path", "")
            if img_path:
                all_paths.append(img_path)
        
        item["image_description"] = " | ".join(all_descriptions)
        item["image_transcription"] = " | ".join(all_transcriptions)
        item["images"] = all_paths
        
        if len(all_paths) > 0:
            images_found += 1
        
        # points processing
        mark_list = []
        for mark_text in marks:
            match = re.search(r'\d+\.?\d*', str(mark_text))
            if match:
                num_str = match.group()
                if "." in num_str:
                    mark_list.append(float(num_str))
                else:
                    mark_list.append(int(num_str))
        
        if len(mark_list) == 1:
            item["points"] = mark_list[0]
        elif len(mark_list) > 1:
            item["points"] = sum(mark_list)
        else:
            item["points"] = None
            
        item.pop("mark", None)
        
        # unique ID creation
        year = item.get("year", "")
        old_id = item.get("id", "")
        school_type = str(item.get("school_type", target_school)).lower()
        item["id"] = f"{new_subj}_{school_type}_{year}_{old_id}"

    logger.info(f"Συνολικά βρέθηκαν {images_found} ερωτήσεις με εικόνες.")

    df = pd.DataFrame(main_dataset)
    df = df.rename(columns={"answer": "answer_text"})
    
    my_columns = [
        "id", "subject", "format", "reference", "question", 
        "input", "images", "choices", "answer_text", "answer_index", 
        "image_description", "image_transcription", "points", "year", "school_type"
    ]
    
    final_columns = [col for col in my_columns if col in df.columns]
    df = df[final_columns]

    # file creation and export
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_file = results_dir / output_filename
    
    df.to_excel(output_file, index=False)
    logger.info(f"✅ Το αρχείο δημιουργήθηκε επιτυχώς στο: {output_file.resolve()}")

    return df

def compare(current_df, reference_file):
    """Συγκρίνει το τρέχον consolidated dataset με ένα reference Excel αρχείο."""
    if current_df.empty:
        logger.error("Δεν μπορεί να γίνει σύγκριση: το current dataset είναι κενό.")
        return

    reference_file = Path(reference_file)

    if not reference_file.exists():
        logger.error(f"Το reference αρχείο δε βρέθηκε: {reference_file}")
        return

    logger.info(f"🔍 Σύγκριση με το αρχείο: {reference_file}")

    ref_df = pd.read_excel(reference_file)

    if "id" not in current_df.columns:
        logger.error("Η στήλη 'id' λείπει από το current dataset.")
        return

    if "id" not in ref_df.columns:
        logger.error("Η στήλη 'id' λείπει από το reference dataset.")
        return

    merged = pd.merge(
        current_df,
        ref_df,
        on="id",
        suffixes=("_cur", "_ref"),
        how="outer",
        indicator=True
    )

    only_cur = merged[merged["_merge"] == "left_only"]
    only_ref = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"]

    logger.info("📊 Στατιστικά σύγκρισης:")
    logger.info(f"   - Match (ίδια ids και στα δύο): {len(both)}")
    logger.info(f"   - Only in Current: {len(only_cur)}")
    logger.info(f"   - Only in Reference: {len(only_ref)}")

    cols_to_check = [
        "subject",
        "format",
        "reference",
        "question",
        "input",
        "choices",
        "answer_text",
        "answer_index",
        "image_description",
        "image_transcription",
        "points",
        "year",
        "school_type"
    ]

    for col in cols_to_check:
        col_cur = f"{col}_cur"
        col_ref = f"{col}_ref"

        if col_cur not in merged.columns or col_ref not in merged.columns:
            logger.warning(f"⚠️ Η στήλη '{col}' δεν υπάρχει και στα δύο datasets. Παραλείπεται.")
            continue

        cur_series = both[col_cur].apply(normalize_for_compare)
        ref_series = both[col_ref].apply(normalize_for_compare)
        
        mismatch = cur_series != ref_series

        if mismatch.any():
            logger.warning(f"Mismatch στη στήλη '{col}': {mismatch.sum()} διαφορές.")

            sample_diffs = both.loc[mismatch, ["id", col_cur, col_ref]].head(5)
            for _, row in sample_diffs.iterrows():
                logger.warning(
                    f"   ID: {row['id']}\n"
                    f"      current  = {row[col_cur]}\n"
                    f"      reference= {row[col_ref]}"
                )
        else:
            logger.info(f"✅ Η στήλη '{col}' ταιριάζει πλήρως.")

    return {
        "only_current": only_cur,
        "only_reference": only_ref,
        "matched": both
    }

def push_to_hub(df, with_images=False, split="train"):
    """Ανεβάζει το processed dataset στο Hugging Face Hub."""
    
    repo_id = os.getenv("HF_REPO_ID")
    token = os.getenv("HF_TOKEN")
    is_private = os.getenv("HF_PRIVATE_REPO", "True").lower() == "true"
    gated_setting = os.getenv("HF_GATED_REPO", "False").lower()

    if not repo_id:
        print("Error: HF_REPO_ID not found in .env")
        return

    if not token:
        print("Error: HF_TOKEN not found in .env")
        return

    print(f"📤 Preparing to push to Hugging Face Hub: {repo_id}")

    try:
        if not repo_exists(repo_id=repo_id, token=token, repo_type="dataset"):
            create_repo(
                repo_id=repo_id,
                token=token,
                private=is_private,
                repo_type="dataset"
            )
            print(f"✅ Created dataset repo: {repo_id}")

        if gated_setting in ["true", "manual"]:
            api = HfApi()
            gated_value = True if gated_setting == "true" else "manual"
            api.update_repo_settings(
                repo_id=repo_id,
                gated=gated_value,
                token=token,
                repo_type="dataset"
            )
            print(f"🔒 Updated gated setting: {gated_value}")

    except Exception as e:
        print(f"⚠️ Error during repo setup: {e}")
        return

    df_hub = df.copy()

    if "answer_index" in df_hub.columns:
        df_hub["answer_index"] = pd.to_numeric(
            df_hub["answer_index"], errors="coerce"
        ).astype("Int64")

    if "points" in df_hub.columns:
        df_hub["points"] = pd.to_numeric(df_hub["points"], errors="coerce")

    for col in ["id", "subject", "format", "reference", "question", "input",
                "answer_text", "image_description", "image_transcription",
                "year", "school_type"]:
        if col in df_hub.columns:
            df_hub[col] = df_hub[col].fillna("").astype(str)

    for col in ["choices", "images"]:
        if col in df_hub.columns:
            df_hub[col] = df_hub[col].apply(
                lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
            )

    # --- Create HF dataset ---
    try:
        dataset = Dataset.from_pandas(df_hub, preserve_index=False)
    except Exception as e:
        print(f"Failed to convert DataFrame to Dataset: {e}")
        return

    # --- Optional image casting ---
    if with_images:
        if "images" in dataset.column_names:
            print("🖼️ Casting 'images' column to Sequence(Image())...")
            try:
                dataset = dataset.cast_column("images", Sequence(Image()))
            except Exception as e:
                print(f"⚠️ Failed to cast 'images' as images: {e}")
                print("ℹ️ Continuing upload without image casting.")
        else:
            print("⚠️ Column 'images' not found, skipping image casting.")

    # --- Push to hub ---
    try:
        dataset.push_to_hub(
            repo_id,
            token=token,
            private=is_private,
            split=split
        )
        print(f"✅ Successfully pushed to Hub (split='{split}').")
    except Exception as e:
        print(f"Failed to push to Hub: {e}")

def main():
    parser = argparse.ArgumentParser(description="Εργαλείο διαχείρισης Dataset Πανελληνίων")
    subparsers = parser.add_subparsers(dest="command", help="Η εντολή που θέλεις να τρέξεις")

    # --- consolidate ---
    con_parser = subparsers.add_parser("consolidate", help="Δημιουργεί το τελικό Excel dataset")
    con_parser.add_argument("--school", default="GEL", help="Τύπος σχολείου (π.χ. GEL)")
    con_parser.add_argument("--output", default="panellinies_dataset.xlsx", help="Όνομα του τελικού Excel")

    # --- compare ---
    cmp_parser = subparsers.add_parser("compare", help="Συγκρίνει το νέο dataset με reference Excel")
    cmp_parser.add_argument("--reference", required=True, help="Το path του reference Excel αρχείου")
    cmp_parser.add_argument("--school", default="GEL", help="Τύπος σχολείου (π.χ. GEL)")
    cmp_parser.add_argument("--output", default="panellinies_dataset.xlsx", help="Όνομα του current Excel που θα δημιουργηθεί")

    # --- push ---
    push_parser = subparsers.add_parser("push", help="Ανεβάζει το dataset στο Hugging Face Hub")
    push_parser.add_argument("--file", type=str, required=True, help="Το Excel αρχείο που θέλεις να ανεβάσεις")
    push_parser.add_argument("--with-images", action="store_true", help="Ενσωμάτωση των πραγματικών εικόνων")
    push_parser.add_argument("--split", type=str, default="train", help="Το target split στο Hugging Face (default: train)")


    args = parser.parse_args()
    
    if args.command == "consolidate":
        df = consolidate(target_school=args.school, output_filename=args.output)
        
    elif args.command == "compare":
        current_df = consolidate(target_school=args.school, output_filename=args.output)
        compare(current_df, args.reference)
        
    elif args.command == "push":
        file_path = Path(args.file)
        if not file_path.exists():
            logger.error(f"Το αρχείο {file_path} δε βρέθηκε!")
            return
            
        df = pd.read_excel(file_path)
        
        for col in ['choices', 'images']:
            if col in df.columns:
                def safe_eval(val):
                    try:
                        if isinstance(val, str) and val.startswith('['):
                            return ast.literal_eval(val)
                        return val
                    except:
                        return val
                df[col] = df[col].apply(safe_eval)
        
        if args.with_images and 'images' in df.columns:
            logger.info("🔍 Αναζήτηση των μονοπατιών για τις εικόνες...")
            
            def fix_image_paths(img_list):
                new_paths = []
                for img in img_list:
                    img_path = Path(img)
                    
                    if img_path.exists():
                        new_paths.append(str(img_path.resolve()))
                        continue
                        
                    found_path = None
                    for p in Path("data").rglob(img_path.name):
                        found_path = str(p.resolve())
                        break 
                        
                    if found_path:
                        new_paths.append(found_path)
                    else:
                        logger.warning(f"⚠️ Η εικόνα δεν βρέθηκε πουθενά: {img}")
                        new_paths.append(img)
                return new_paths
                
            df['images'] = df['images'].apply(fix_image_paths)
                
        push_to_hub(df, with_images=args.with_images, split=args.split)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

#Πώς δουλεύουν πλέον οι εντολές στο τερματικό:
#Για να φτιάξεις το Excel:
#uv run src/build_dataset.py consolidate

#Για να ελέγξεις αν κάτι χάλασε σε σχέση με χθες:
#uv run src/build_dataset.py compare --reference ../παλιό_αρχείο.xlsx

#Για να το στείλεις στο Hugging Face (μαζί με τις φωτογραφίες):
#uv run src/build_dataset.py push --file results/panellinies_dataset.xlsx --with-images
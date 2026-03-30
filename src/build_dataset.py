import os
import json
import re
import pandas as pd
import argparse
from pathlib import Path
import ast
import random
import logging
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
def get_file_pairs(data_dir):
    data_dir = Path(data_dir)
    pairs = []
    
    for json_path in data_dir.rglob("*.json"):
        json_filename = json_path.name
        md_filename = json_filename.replace("them_", "apant_").replace(".json", ".md")
        
        md_path = json_path.parent / md_filename
        
        if md_path.exists():
            pairs.append({"json": json_path, "md": md_path})
        else:
            logger.warning(f"Βρέθηκε το {json_filename} χωρίς απάντηση.")
            
    return pairs

# --- DATA TRANSFORMERS ---        
#def detect_exercise_type
#def unify_label
#def find_answer_index
#def extract_points
#def apply_structural_tags 
#def parse_image_txt

# --- DATA CONSOLIDATION ---
#def consolidate


#Suggested columns for final df:
#id
#subject
#format (multiple_choice, true_false, matching, fill_in_the_gaps, open_ended)
#reference (none, passage, multimodal, table)
#question
#input
#images
#choices
#answer_text
#answer_index
#image_description
#image_transcription
#points
#year
#admission_level
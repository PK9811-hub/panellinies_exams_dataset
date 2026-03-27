import os
import json
import re
import pandas as pd
import argparse
from pathlib import Path
import ast
import random
from datasets import Dataset, Image, Features, Sequence
from huggingface_hub import create_repo, repo_exists, HfApi
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# --- HELPER FUNCTIONS ---

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

def parse_markdown_answers(md_path):
    answers_dict = {}
    md_path = Path(md_path)
    
    if not md_path.exists():
        print(f"Προσοχή: Το αρχείο {md_path} δεν βρέθηκε.")
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
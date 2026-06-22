import ast
import random
import pandas as pd
import os
import logging
import traceback
from datasets import load_dataset, concatenate_datasets
from dotenv import load_dotenv, find_dotenv
import lm_eval
from lm_eval.models.openai_completions import OpenAIChatCompletion
from lm_eval.models.huggingface import HFLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_matching_row(row):
    """
    Processes a single matching exercise row to create a list of answer options
    (shuffled versions of the matching) and identifying the correct index."""
    correct_answer = row.get("answer_text", "")
    
    if not correct_answer:
        return None
    try:
        matching_answer = ast.literal_eval(correct_answer)
    except:
        return None
    
    left_items = []
    right_items = []
    
    try:
        for pair in matching_answer:
            pieces = pair.split("-")
            left_items.append(pieces[0])
            right_items.append(pieces[1])
    except:
        return None
    
    distractors = []
    max_attempts = 100
    while len(distractors) < 3 and max_attempts > 0:
        shuffled_right = random.sample(right_items, len(right_items))
        
        candidate=[]
        for i in range(len(left_items)):
            left = left_items[i]
            right = shuffled_right[i]
            new_pair = left + "-" + right
            candidate.append(new_pair)
    
        max_attempts -= 1
        
        if candidate != matching_answer:
            if candidate not in distractors:
                distractors.append(candidate)
    
    all_options = [matching_answer]
    for distractor in distractors:
        all_options.append(distractor)
        
    random.shuffle(all_options)

    final_choices = []
    for option in all_options:
        text_option = "[" + ", ".join(option) + "]"
        final_choices.append(text_option)
    
    correct_text_option = "[" + ", ".join(matching_answer) + "]"
    correct_index = final_choices.index(correct_text_option)
    
    return pd.Series({
        "processed_choices" : final_choices,
        "new_answer_index" : correct_index
    })
                
def apply_matching_processing(df):
    df = df.copy()
    
    df['processed_choices'] = None
    df['new_answer_index'] = None
    
    for index, row in df.iterrows():
        if row['format'] == "matching":
            result = process_matching_row(row)
            
            if result is not None:
                df.at[index, "processed_choices"] = result["processed_choices"]
                df.at[index, "new_answer_index"] = result["new_answer_index"]
    
    return df

def load_panellinies_exams_dataset(repo_id=None, split=None):
    """
    Loads the dataset. If no split is specified, returns the concatenated train and test sets.
    """
    if repo_id is None:
        repo_id = os.getenv("HF_REPO_ID")
    
    logger.info(f"Loading dataset from Hugging Face: {repo_id}")
    dataset_dict = load_dataset(repo_id)
    
    if split:
        if split in dataset_dict:
            return dataset_dict[split]
        else:
            logger.error((f"Split '{split}' not found in {repo_id}"))
            return None
    
    available_splits = list(dataset_dict.keys)
    
    if len(available_splits) == 1:
        return dataset_dict[available_splits[0]]
    
    logger.info(f"Concatenating splits: {available_splits}")
    return concatenate_datasets([dataset_dict[s] for s in available_splits])

def run_evaluation(model_name, backend="api", api_base=None, task_dict=None, eval_limit=None):
    """
    Runs evaluation.
    - backend="api": For API models (KriKri web service, OpenAI, etc).
    - backend="hf": For open models (Hugging Face) which run locally on GPU.
    """
    logger.info(f"Starting evaluation for model: {model_name} (Backend: {backend})")
    
    try:
        if backend == "api":
            # ---------------------------------------------------------
            # Scenario 1: API call
            # ---------------------------------------------------------
            chat_api_url = api_base or os.getenv("OPENAI_BASE_URL")
            
            if not chat_api_url:
                raise ValueError("No API base URL provided. Set OPENAI_BASE_URL in .env.")

            if not chat_api_url.endswith("/chat/completions"):
                chat_api_url = chat_api_url.rstrip("/") + "/chat/completions"

            model = OpenAIChatCompletion(
                model=model_name,
                base_url=chat_api_url,
                num_fewshot=0,
                eos_string="<|end_of_text|>",
                max_retries=10,
                num_concurrent=1
            )
            
        elif backend == "hf":
            # ---------------------------------------------------------
            # Scenario 2: Local call on GPU server
            # ---------------------------------------------------------
            model = HFLM(
                pretrained=model_name,
                device="cuda",  
                batch_size="auto"
            )
            
        else:
            raise ValueError(f"Unsupported backend: {backend}. Use 'api' or 'hf'.")

        results = lm_eval.evaluate(
            lm=model,
            task_dict=task_dict,
            limit=eval_limit,
            apply_chat_template=True
        )
        return results

    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        logger.error(traceback.format_exc())
        return None
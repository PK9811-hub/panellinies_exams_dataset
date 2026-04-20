import ast
import random
import pandas as pd

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

#functions to be added from protipa exams dataset

#def filter_dataset()

#def load_panellinies_dataset()

#def clean_dataset_paths()

#def process_results_open() ?
#def process_results_bypass() ?

#def run_evaluation()
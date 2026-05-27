import datasets
import re
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False

class GreekTokenizer:
    def tokenize(self, text):
        text = text.lower()
        return re.findall(r'[a-z0-9\u0370-\u03ff\u1f00-\u1fff]+', text)


ROUGE_SCORER = None


def process_results_gen(doc, results):
    completion = results[0]
    
    gold_answer = doc.get("answer_text") or doc.get("answer") or ""
    true_refs = [gold_answer]

    # BLEU (sacrebleu with international tokenization)
    bleu_scores = [bleu([[ref]], [completion]) for ref in true_refs]
    bleu_max = np.nanmax(bleu_scores)

    # ROUGE with custom Greek-safe tokenizer
    global ROUGE_SCORER
    if ROUGE_SCORER is None:
        ROUGE_SCORER = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeLsum"], 
            tokenizer=GreekTokenizer()
        )
    
    rouge_scores = []
    for ref in true_refs:
        scores = ROUGE_SCORER.score(ref, completion)
        rouge_scores.append({
            "rouge1": scores["rouge1"].fmeasure * 100.0,
            "rouge2": scores["rouge2"].fmeasure * 100.0,
            "rougeL": scores["rougeLsum"].fmeasure * 100.0,
        })

    rouge1_max = np.nanmax([s["rouge1"] for s in rouge_scores])
    rouge2_max = np.nanmax([s["rouge2"] for s in rouge_scores])
    rougeL_max = np.nanmax([s["rougeL"] for s in rouge_scores])

    # BERTScore using multilingual BERT to handle semantic similarity in Greek
    bertscore_f1_max = 0.0
    if BERTSCORE_AVAILABLE:
        # P, R, F1 are returned as tensors
        P, R, F1 = bert_score_fn(
            [completion],
            true_refs,
            lang="el",
            model_type="bert-base-multilingual-cased",
            verbose=False,
        )
        bertscore_f1_max = F1.max().item()

    return {
        "bleu_max": bleu_max,
        "rouge1_max": rouge1_max,
        "rouge2_max": rouge2_max,
        "rougeL_max": rougeL_max,
        "bertscore_f1_max": bertscore_f1_max,
    }


def bleu(refs, preds):
    score = sacrebleu.corpus_bleu(
        preds,
        refs,
        smooth_method="exp",
        smooth_value=0.0,
        force=False,
        lowercase=False,
        tokenize="intl",
        use_effective_order=False,
    ).score
    return score



def filter_by_mode_and_subject(dataset, mode='closed', subject=None):
    """
    Unified filter for evaluation mode and subject.
    - Closed Mode: MCQ, T/F, Matching, and Fill-in-the-gaps with choices.
    - Open Mode: Open-ended and Fill-in-the-gaps without choices.
    - Subject: Optional filtering by subject string.
    """
    def _filter_logic(x):
        # Optional subject filter 
        if subject and x.get("subject", "").lower().strip() != subject.lower().strip():
            return False
            
        fmt = x.get("format")
        choices = x.get("choices")
        has_choices = isinstance(choices, list) and len(choices) > 0
        
        if mode == 'closed':
            return fmt in ["multiple_choice", "true_false", "matching"] or \
                   (fmt == "fill_in_the_gaps" and has_choices)
        elif mode == 'open':
            return fmt == "open_ended" or \
                   (fmt == "fill_in_the_gaps" and not has_choices)
        return True

    return dataset.filter(_filter_logic)

def doc_to_text_closed(doc):
    """
    MCQ prompt logic with specific Greek instructions for index-based answers.
    """
    prompt_parts = []
    
    # Contextual Inputs
    if doc.get("input"): 
        prompt_parts.append(doc["input"])
    if doc.get("image_description"): 
        prompt_parts.append(f"Περιγραφή εικόνας: {doc['image_description']}")
    if doc.get("image_transcription"): 
        prompt_parts.append(f"Κείμενο εικόνας: {doc['image_transcription']}")
    
    # Question
    prompt_parts.append(f"Ερώτηση: {doc['question']}")
    
    # Choices (Index-based)
    if doc.get("choices"):
        prompt_parts.append("Επιλογές:")
        for i, choice in enumerate(doc["choices"]):
            prompt_parts.append(f"{i}. {choice}")
            
    # Critical Instructions
    instruction = (
        "\n### ΟΔΗΓΙΑ ΜΟΡΦΟΠΟΙΗΣΗΣ (CRITICAL)\n"
        "Πρέπει να παρέχεις ΜΟΝΟ τον αριθμό του δείκτη (index) της σωστής επιλογής (π.χ. 0, 1, 2, 3...).\n"
        "ΠΡΟΣΟΧΗ: Ο αριθμός '2' που χρησιμοποιείται στα παραδείγματα παρακάτω είναι ΤΥΧΑΙΟΣ και αφορά μόνο τη ΜΟΡΦΗ της απάντησης εδώ.\n"
        "Η σωστή απάντηση εξαρτάται αποκλειστικά από την ερώτηση και μπορεί να είναι ΟΠΟΙΟΣΔΗΠΟΤΕ αριθμός.\n"
        "Μην επεξηγείς και μην γράφεις ολόκληρες προτάσεις.\n\n"
        "Παραδείγματα Μορφής:\n"
        "❌ ΛΑΘΟΣ: \"Η σωστή επιλογή είναι η 2.\"\n"
        "❌ ΛΑΘΟΣ: \"(2)\"\n"
        "✅ ΣΩΣΤΟ: 2 (ή 0 ή 1 ή 3... ανάλογα με τη σωστή επιλογή)\n\n"
        "Απάντηση:"
    )
    prompt_parts.append(instruction)
    return "\n".join(prompt_parts)

def doc_to_target(doc):
    """Extracts the integer index as the target string."""
    if doc.get("answer_index") is not None:
        return str(doc["answer_index"]).split(',')[0].strip()
    return ""

def doc_to_text_open(doc):
    """
    Prompt logic for open-ended and fill-in-the-gap questions.
    """
    prompt_parts = []
    
    # Instruction
    instruction = (
        "Δίνεται η παρακάτω ερώτηση από σχολικές εξετάσεις.\n"
        "Αν είναι ερώτηση ανάπτυξης, δώσε μια ολοκληρωμένη και τεκμηριωμένη απάντηση.\n"
        "Αν είναι ερώτηση συμπλήρωσης κενών, γράψε τη σωστή λέξη ή τη σωστή φράση που λείπει."
    )
    prompt_parts.append(instruction)
    
    # Contextual Inputs 
    if doc.get("input"): 
        prompt_parts.append(f"Πλαίσιο/Κείμενο: {doc['input']}")
    if doc.get("image_description"): 
        prompt_parts.append(f"Περιγραφή εικόνας: {doc['image_description']}")
    if doc.get("image_transcription"): 
        prompt_parts.append(f"Κείμενο εικόνας: {doc['image_transcription']}")
    
    # Question
    prompt_parts.append(f"Ερώτηση: {doc['question']}\n\nΑπάντηση:")
    
    return "\n\n".join([p for p in prompt_parts if p.strip()])

def doc_to_target_open(doc):
    """Extracts the expected text answer for open-ended evaluation."""
    ans = doc.get("answer_text") or doc.get("answer") or ""
    return [str(ans).strip()]


# =====================================================================
# WRAPPER FUNCTIONS 
# =====================================================================

# --- CLOSED TASKS (Multiple Choice, True/False, Matching) ---
def process_ancient_greek_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='ancient_greek')
def process_biology_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='biology')
def process_chemistry_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='chemistry')
def process_computer_science_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='computer_science')
def process_economics_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='economics')
def process_greek_language_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='greek_language')
def process_history_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='history')
def process_latin_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='latin')
def process_mathematics_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='mathematics')
def process_physics_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='physics')

# --- OPEN TASKS (Open-ended, Fill-in-the-gaps) ---
def process_ancient_greek_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='ancient_greek')
def process_biology_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='biology')
def process_chemistry_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='chemistry')
def process_computer_science_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='computer_science')
def process_economics_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='economics')
def process_greek_language_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='greek_language')
def process_history_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='history')
def process_latin_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='latin')
def process_mathematics_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='mathematics')
def process_physics_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='physics')
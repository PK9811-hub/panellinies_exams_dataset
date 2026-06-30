import datasets
import re
import numpy as np
import sacrebleu
from rouge_score import rouge_scorer
from bert_score import BERTScorer

from bert_score import score as bert_score_fn

bert_scorer = BERTScorer(lang="el",  model_type="bert-base-multilingual-cased")


class GreekTokenizer:
    def tokenize(self, text):
        text = text.lower()
        return re.findall(r'[a-z0-9\u0370-\u03ff\u1f00-\u1fff]+', text)


ROUGE_SCORER = None


def process_results_gen(doc, results):
    completion = results[0]
    completion = re.sub(r'[*()\[\]"\']', '', completion).strip()
    
    gold_answer = doc.get("answer_text") or doc.get("answer") or ""
    gold_answer = str(gold_answer).strip()
    
    if "/" in gold_answer:
        true_refs = [a.strip() for a in gold_answer.split("/")]
    else:
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

    # P, R, F1 are returned as tensors
    P, R, F1 = bert_scorer.score(
        [completion]* len(true_refs),
        true_refs,
        # lang="el",
        # model_type="bert-base-multilingual-cased",
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
    - Closed Mode: MCQ, T/F, and Fill-in-the-gaps with choices.
    - Structured Mode: Matching, and Fill-in-the-gaps without choices.
    - Open Mode: Open-ended (free text / essays).
    - Subject: Optional filtering by subject string.
    """
    def _filter_logic(x):
        # Optional subject filter
        if subject and x.get("subject") != subject:
            return False
            
        fmt = x.get("format")
        choices = x.get("choices")
        has_choices = isinstance(choices, list) and len(choices) > 0
        
        if mode == 'closed':
            return fmt in ["multiple_choice", "true_false"] or \
                   (fmt == "fill_in_the_gaps" and has_choices)
        elif mode == 'structured':
            return fmt == "matching" or \
                   (fmt == "fill_in_the_gaps" and not has_choices)
        elif mode == 'open':
            return fmt == "open_ended"
        return True

    return dataset.filter(_filter_logic)

def doc_to_text_closed(doc):
    """
    MCQ prompt logic with specific Greek instructions for index-based answers.
    """
    prompt_parts = []
    
    # 1. Contextual Inputs
    if doc.get("input"): 
        prompt_parts.append(doc["input"])
    if doc.get("image_description"): 
        prompt_parts.append(f"Περιγραφή εικόνας: {doc['image_description']}")
    if doc.get("image_transcription"): 
        prompt_parts.append(f"Κείμενο εικόνας: {doc['image_transcription']}")
    
    # 2. Question
    prompt_parts.append(f"Ερώτηση: {doc['question']}")
    
    # 3. Choices (Index-based)
    if doc.get("choices"):
        prompt_parts.append("Επιλογές:")
        for i, choice in enumerate(doc["choices"]):
            # Fix choice-stripping bug: if rest of choice is empty, keep the original letter label
            match = re.match(r'^([Α-Ωα-ωA-Za-z0-9]+)\.\s*(.*)$', str(choice))
            if match:
                letter, rest = match.groups()
                cleaned_choice = rest.strip() if rest.strip() else letter
            else:
                cleaned_choice = str(choice).strip()
            prompt_parts.append(f"{i}. {cleaned_choice}")
            
    # 4. Critical Instructions
    instruction = (
        "\nΕίσαι ένας 18χρονος τελειόφοιτος Λυκείου που απαντά σε διαγώνισμα Πανελλαδικών.\n"
        "\nΑπάντησε ΜΟΝΟ με τον αριθμό της σωστής επιλογής (δηλαδή 0, 1, 2 ή 3).\n"
        "Μην γράφεις καμία άλλη λέξη, επεξήγηση ή σημεία στίξης.\n"
        "Απάντηση:"
    )
    prompt_parts.append(instruction)
    return "\n".join(prompt_parts)

def doc_to_target_index(doc):
    """Extracts target index directly from answer_index field."""
    return str(doc["answer_index"]).split(',')[0].strip()

def doc_to_target_true_false(doc):
    """Maps true_false text answer back to its choices index."""
    choices = doc.get("choices")
    answer_text = doc.get("answer_text") or doc.get("answer")
    if choices and answer_text:
        answer_text_clean = str(answer_text).strip().lower()
        for i, choice in enumerate(choices):
            choice_str = str(choice).strip().lower()
            # Remove leading label prefix like 'α.', 'β.', '1.', 'a.'
            choice_clean = re.sub(r'^[α-ωa-z0-9]+\.\s*', '', choice_str).strip()
            if answer_text_clean == choice_clean or answer_text_clean in choice_str:
                return str(i)
    return ""

def doc_to_target(doc):
    """Extracts the integer index as the target string."""
    if doc.get("answer_index") is not None:
        return doc_to_target_index(doc)
    
    if doc.get("format") == "true_false":
        return doc_to_target_true_false(doc)
        
    return ""

def doc_to_text_open(doc):
    """
    Prompt logic for open-ended and fill-in-the-gap questions.
    """
    prompt_parts = []
    format_type = doc.get("format")
    
    # 1. Instruction Engineering 
    if format_type == "open_ended":
        instruction = (
    "\nΕίσαι ένας 18χρονος τελειόφοιτος Λυκείου που απαντά σε διαγώνισμα Πανελλαδικών.\n"
    "Απάντησε στην παρακάτω ερώτηση ανάπτυξης, δίνοντας μια ολοκληρωμένη και τεκμηριωμένη απάντηση.\n"
    "ΠΡΟΣΟΧΗ: Ξεκίνα την απάντησή σου απευθείας, χωρίς εισαγωγικές φράσεις, χωρίς να επαναλάβεις την ερώτηση και χωρίς χαιρετισμούς."
    )
        
    elif format_type == "fill_in_the_gaps":
        instruction = (
    "\nΕίσαι ένας 18χρονος τελειόφοιτος Λυκείου που απαντά σε διαγώνισμα Πανελλαδικών.\n"
    "Γράψε ΜΟΝΟ τη σωστή λέξη ή τη σωστή φράση/τύπο που λείπει στην ερώτηση συμπλήρωσης κενών που σου δίνεται.\n"
    "ΚΡΙΣΙΜΗ ΟΔΗΓΙΑ: Μην δίνεις καμία απολύτως εξήγηση, μην γράφεις ολόκληρες προτάσεις, και μην χρησιμοποιείς εισαγωγικά.\n"
    "Η απάντησή σου πρέπει να περιέχει αποκλειστικά και ΜΟΝΟ τη λέξη ή φράση που συμπληρώνει το κενό."
    )
    
    prompt_parts.append(instruction)
    if doc.get("input"): 
        prompt_parts.append(f"Πλαίσιο/Κείμενο: {doc['input']}")
    if doc.get("image_description"): 
        prompt_parts.append(f"Περιγραφή εικόνας: {doc['image_description']}")
    if doc.get("image_transcription"): 
        prompt_parts.append(f"Κείμενο εικόνας: {doc['image_transcription']}")
    
    # 2. Question
    prompt_parts.append(f"Ερώτηση: {doc['question']}\n\nΑπάντηση:")
    
    return "\n\n".join(prompt_parts)

def doc_to_target_open(doc):
    """Extracts the expected text answer for open-ended evaluation."""
    ans = doc.get("answer_text") or doc.get("answer") or ""
    ans_str = str(ans).strip()

    if "/" in ans_str:
        return [a.strip() for a in ans_str.split("/")]
    
    return [ans_str]


# ==========================================

# ------ CLOSED MODE (MCQ, T/F) ------
def process_ancient_greek_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='ancient_greek')
def process_economics_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='economics')
def process_physics_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='physics')
def process_history_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='history')
def process_latin_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='latin')
def process_mathematics_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='mathematics')
def process_greek_language_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='greek_language')
def process_computer_science_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='computer_science')
def process_chemistry_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='chemistry')
def process_biology_closed(dataset): return filter_by_mode_and_subject(dataset, mode='closed', subject='biology')

# ------ STRUCTURED MODE (Matching, Fill-in-gaps) ------
def process_ancient_greek_structured(dataset): return filter_by_mode_and_subject(dataset, mode='structured', subject='ancient_greek')
def process_latin_structured(dataset): return filter_by_mode_and_subject(dataset, mode='structured', subject='latin')
def process_computer_science_structured(dataset): return filter_by_mode_and_subject(dataset, mode='structured', subject='computer_science')
def process_biology_structured(dataset): return filter_by_mode_and_subject(dataset, mode='structured', subject='biology')

# ------ OPEN MODE (Ελεύθερης ανάπτυξης) ------
def process_ancient_greek_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='ancient_greek')
def process_economics_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='economics')
def process_physics_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='physics')
def process_history_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='history')
def process_latin_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='latin')
def process_mathematics_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='mathematics')
def process_greek_language_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='greek_language')
def process_computer_science_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='computer_science')
def process_chemistry_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='chemistry')
def process_biology_open(dataset): return filter_by_mode_and_subject(dataset, mode='open', subject='biology')


def doc_to_text_matching(doc):
    """Formats the prompt specifically for matching format questions."""
    prompt_parts = []
    if doc.get("input"): 
        prompt_parts.append(doc["input"])
    if doc.get("image_description"): 
        prompt_parts.append(f"Περιγραφή εικόνας: {doc['image_description']}")
    if doc.get("image_transcription"): 
        prompt_parts.append(f"Κείμενο εικόνας: {doc['image_transcription']}")
    
    prompt_parts.append(f"Ερώτηση: {doc['question']}")
    
    instruction = (
        "\nΕίσαι ένας 18χρονος τελειόφοιτος Λυκείου που απαντά σε διαγώνισμα Πανελλαδικών.\n"
        "\nΑντιστοίχισε τα στοιχεία της αριστερής στήλης με αυτά της δεξιάς στήλης.\n"
        "Γράψε την απάντησή σου ΜΟΝΟ στη μορφή: 1-α, 2-β... (ή Α-α, Β-β... ανάλογα με τα στοιχεία των στηλών).\n"
        "Μην γράφεις καμία άλλη λέξη, επεξήγηση ή πίνακα.\n"
        "Απάντηση:"
    )
    prompt_parts.append(instruction)
    return "\n".join(prompt_parts)

def doc_to_target_matching(doc):
    """Extracts expected matching pairings."""
    ans = doc.get("answer_text") or doc.get("answer") or ""
    return str(ans).strip()

def doc_to_text_structured(doc):
    """Dispatches prompt generation for structured mode formats (matching and open fill-in-gaps)."""
    fmt = doc.get("format")
    if fmt == "matching":
        return doc_to_text_matching(doc)
    
    # For fill-in-the-gaps (no choices)
    instruction = (
        "\nΕίσαι ένας 18χρονος τελειόφοιτος Λυκείου που απαντά σε διαγώνισμα Πανελλαδικών.\n"
        "Γράψε ΜΟΝΟ τη σωστή λέξη ή τη σωστή φράση/τύπο που λείπει στην ερώτηση συμπλήρωσης κενών που σου δίνεται.\n"
        "ΚΡΙΣΙΜΗ ΟΔΗΓΙΑ: Μην δίνεις καμία απολύτως εξήγηση, μην γράφεις ολόκληρες προτάσεις, και μην χρησιμοποιείς εισαγωγικά.\n"
        "Η απάντησή σου πρέπει να περιέχει αποκλειστικά και ΜΟΝΟ τη λέξη ή φράση που συμπληρώνει το κενό."
    )
    prompt_parts = [instruction]
    if doc.get("input"): 
        prompt_parts.append(f"Πλαίσιο/Κείμενο: {doc['input']}")
    if doc.get("image_description"): 
        prompt_parts.append(f"Περιγραφή εικόνας: {doc['image_description']}")
    if doc.get("image_transcription"): 
        prompt_parts.append(f"Κείμενο εικόνας: {doc['image_transcription']}")
    
    prompt_parts.append(f"Ερώτηση: {doc['question']}\n\nΑπάντηση:")
    return "\n\n".join(prompt_parts)

def doc_to_target_structured(doc):
    """Extracts expected targets for structured mode formats."""
    fmt = doc.get("format")
    if fmt == "matching":
        return doc_to_target_matching(doc)
    
    # For fill-in-the-gaps (no choices)
    ans = doc.get("answer_text") or doc.get("answer") or ""
    return str(ans).strip()

def parse_matching_pairs(text):
    """Parses strings like '1-γ, 2-α' into a normalized set of sorted tuples."""
    pairs = re.findall(r'([a-zA-Zα-ωΑ-Ω0-9]+)\s*-\s*([a-zA-Zα-ωΑ-Ω0-9]+)', str(text))
    normalized_pairs = set()
    for p1, p2 in pairs:
        sorted_pair = tuple(sorted([p1.strip().lower(), p2.strip().lower()]))
        normalized_pairs.add(sorted_pair)
    return normalized_pairs

def matching_accuracy_metric(references, predictions):
    """Calculates Jaccard overlap of correct pairs (0.0 to 1.0)."""
    pred_text = predictions[0] if predictions else ""
    ref_text = references[0] if references else ""
    
    pred_pairs = parse_matching_pairs(pred_text)
    ref_pairs = parse_matching_pairs(ref_text)
    
    if not ref_pairs:
        return 0.0
    
    correct_matches = len(pred_pairs.intersection(ref_pairs))
    return correct_matches / len(ref_pairs)

def clean_greek_text(text):
    """Normalizes Greek text by lowercasing and stripping accents/tonos."""
    text = str(text).strip().lower()
    accents_map = {
        'ά': 'α', 'έ': 'ε', 'ή': 'η', 'ί': 'ι', 'ό': 'ο', 'ύ': 'υ', 'ώ': 'ω',
        'ϊ': 'ι', 'ϋ': 'υ', 'ΐ': 'ι', 'ΰ': 'υ'
    }
    for acc, plain in accents_map.items():
        text = text.replace(acc, plain)
    return text

def structured_short_answer_metric(references, predictions):
    """Evaluates open fill-in-the-gaps using normalized exact match (0.0 or 1.0)."""
    pred_clean = clean_greek_text(predictions[0] if predictions else "")
    ref_clean = clean_greek_text(references[0] if references else "")
    return 1.0 if pred_clean == ref_clean else 0.0

def process_results_structured(doc, results):
    """Processes results for structured mode tasks using a unified accuracy metric."""
    fmt = doc.get("format")
    pred = results[0] if results else ""
    target = doc_to_target_structured(doc)
    
    if pred:
        pred = pred.replace("**", "").replace("_", "")
        parts = re.split(r'\\n|\n|\r\n|\s{2,}', pred)
        pred = parts[0]
        pred = pred.split("(")[0].split("[")[0]
        pred = re.split(r'(Εξήγηση|Σημείωση|Παρατηρήσεις|Γράψε)', pred)[0]
        pred = pred.strip()

    if fmt == "matching":
        score = matching_accuracy_metric([target], [pred])
    else:
        score = structured_short_answer_metric([target], [pred])
        
    return {
        "structured_accuracy": score
    }
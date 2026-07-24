import sys
from collections import defaultdict
from inspect_ai.log import read_eval_log

def aggregate_scores(log_path):
    print(f"Reading log file: {log_path}")
    try:
        log = read_eval_log(log_path)
    except Exception as e:
        print(f"Error reading log file: {e}")
        sys.exit(1)
        
    if not log.samples:
        print("No samples found in the log file.")
        return

    # Check first sample metadata
    first_sample = log.samples[0]
    print("\nMetadata keys available in samples:", list(first_sample.metadata.keys()) if first_sample.metadata else [])

    # Group scores by subject and scorer
    subject_scores = defaultdict(list)
    for sample in log.samples:
        metadata = sample.metadata or {}
        subject = metadata.get("subject", "Unknown")
        
        if sample.scores:
            for scorer_name, score_obj in sample.scores.items():
                val = score_obj.value
                try:
                    val_float = float(val)
                    subject_scores[(subject, scorer_name)].append(val_float)
                except (ValueError, TypeError):
                    pass

    # Print results as Markdown table
    print("\n| Subject | Scorer | Count | Average Score |")
    print("| :--- | :--- | :---: | :---: |")
    for (subj, scorer_name), scores in sorted(subject_scores.items()):
        avg = sum(scores) / len(scores) if scores else 0.0
        print(f"| {subj} | {scorer_name} | {len(scores)} | {avg:.4f} |")


if __name__ == "__main__":
    log_file = ".agents/results/2026-07-24_qwen3-32b_open_ended_0-shot_t_0.1_top_p_0.9_top_k_40.eval"
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    aggregate_scores(log_file)

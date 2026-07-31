# Pan-Ex Dataset

Pan-Ex ([Link withheld for double-blind review]) is a comprehensive dataset of Greek Panhellenic Exams for NLP research and educational analysis. Includes structured data (JSON-questions & MD-answers), raw files, and processing scripts. It is a dataset derived from publicly available exam questions and official solutions used for student admission to Higher Education Institutions in Greece via the Panhellenic Examinations.

The dataset includes questions with the following features:

* **Subjects:** Greek Language, Ancient Greek, History, Latin, Biology, Physics, Chemistry, Computer Science, Economics, and Mathematics.
* **Educational Levels:** Targeted at General Lyceum (Γενικό Λύκειο - GEL) graduates.
* **Formats:** Multiple Choice, True/False, Matching, Fill-in-the-Gaps, and Open-Ended questions.
* **Modalities:** Questions suitable for multimodal evaluation, featuring high-fidelity images/diagrams, LLM-generated image descriptions, and OCR transcriptions.

The benchmark can be used for the evaluation of LLMs on complex, multi-subject, multi-format questions in the Greek language. Additionally, it may be useful as a high-quality resource for quantitative educational research.

## Dataset Creation

The source material was extracted from the official portals of the Greek Ministry of Education. It was then converted into the current format via specialized processing pipelines by the authors.

**Disclaimer:** While every effort has been made to ensure the accuracy and completeness of this structured dataset, any errors, omissions, or formatting issues are the result of the processing and transformation pipeline and are *not related* to the original source or the Ministry of Education.

## Dataset Structure

| Column                  | Description                                                            |
| :---------------------- | :--------------------------------------------------------------------- |
| `id`                  | Unique identifier (e.g.,`physics_gel_2024_B1`).                      |
| `subject`             | Academic subject (e.g.,`physics`, `chemistry`, `ancient_greek`). |
| `format`              | Extracted format (e.g.,`multiple_choice`, `open_ended`).           |
| `reference`           | Additional reference inputs (e.g., passage, multimodal, table).        |
| `question`            | The core question text.                                                |
| `input`               | Additional context or passage required to answer the question.         |
| `images`              | List of local paths to visual asset(s) (diagrams, photos).             |
| `choices`             | Candidate answers for closed-ended questions (List).                   |
| `answer_text`         | The correct answer/solution text.                                      |
| `answer_index`        | The index of the correct answer for multiple-choice questions.         |
| `image_description`   | LLM-generated textual descriptions of visual assets.                   |
| `image_transcription` | OCR/Text extraction from within the visual assets.                     |
| `points`              | Assigned point value for the question. May be null if missing.         |
| `year`                | The year of the exam.                                                  |
| `school_type`         | The target education level/school type (e.g.,`gel`).                 |

## Usage

```python
import random

# Load a random sample
random_idx = random.randint(0, len(dataset) - 1)
sample = dataset[random_idx]

print(f"Sample Index: {random_idx} | ID: {sample['id']}")
print(f"Question: {sample['question']}\n")

# 1. Handle Multimodal Metadata (Descriptions & Transcriptions)
if sample.get('image_description'):
    print(f"🖼️  Image Description: {sample['image_description']}")
if sample.get('image_transcription'):
    print(f"📝 Image Transcription: {sample['image_transcription']}")

# 2. Handle Choices
if sample.get('choices'):
    print("\nChoices:")
    correct_idx = sample.get('answer_index')
    for i, choice in enumerate(sample['choices']):
        # Handle cases where answer_index might be missing
        marker = "[✅]" if correct_idx is not None and i == int(correct_idx) else "[  ]"
        print(f"  {marker} {i}: {choice}")
    print(f"\nCorrect Answer Index: {correct_idx}")
else:
    print(f"\nAnswer: {sample['answer_text']}")
```

## Benchmarking & Evaluation

To facilitate the seamless evaluation of Large Language Models, this repository provides ready-to-use configurations for two popular evaluation frameworks:

### 1. LM Evaluation Harness (`lm-eval`)

All tasks compatible with the EleutherAI `lm-eval` harness can be found in the [`tasks/panellinies`](https://github.com/PK9811-hub/panellinies_exams_dataset/tree/main/tasks/panellinies) directory.

* You can run experiments across different overarching formats, including **open-ended**, **closed-ended**, and **structured aggregate** (short phrases/single words) exercises.
* Individual tasks are also broken down by specific academic subjects and their corresponding question types, allowing for highly targeted benchmarking.

### 2. Inspect AI (LLM-as-a-Judge)

For evaluating complex, open-ended questions where standard exact-match metrics fall short, we utilize the **Inspect AI** framework employing an LLM-as-a-judge methodology.

* The necessary Python evaluation scripts, along with the prompt configurations (in the `configs` folder), are located in the [`src/evals`](https://github.com/PK9811-hub/panellinies_exams_dataset/tree/main/src/evals) directory.

#### Running Evaluations

<details>
<summary>Running Inspect AI Evaluations</summary>

Make sure your `.env` file is set up with the correct variables. First, load the environment variables:

```bash
export $(grep -v '^#' .env | xargs)
```

The evaluation script is highly flexible. You can evaluate the entire dataset, or filter it down using standard inspect ai flags (like --limit for quick testing) and custom task parameters (-T).

Filtering Rules:

* Use -T filter_field and -T filter_value to specify metadata columns and their allowed values.

* Use ; to apply multiple filters (AND logic) (e.g., -T filter_field="subject;format").

* Use , to allow multiple values for a single field (OR logic) (e.g., -T filter_value="physics;open_ended,fill_in_the_gaps").

* Use -T filter_field="has_image_description" -T filter_value="true" to only evaluate questions that contain image descriptions.

Here are some examples of how to run the evaluations:

**Example 1: Specific Subject and Format (Open-ended Greek Language)**

```bash
uv run inspect eval src/evals/tasks.py \
  --model "openai/$MODEL_ID" \
  --max-connections 10 \
  -T dataset_path="$HF_REPO_ID" \
  -T split=train \
  -T input_field=question \
  -T target_field=answer_text \
  -T filter_field="format;subject" \
  -T filter_value="open_ended;greek_language" \
  -T grader_model="openai/$GRADER_MODEL_ID" \
  --batch false \
  -M responses_api=false
```

**Example 2: Multiple Formats & Limit Samples (Quick Test in Biology)**

```bash
uv run inspect eval src/evals/tasks.py \
  --model "openai/$MODEL_ID" \
  --limit 5 \
  --max-connections 10 \
  -T dataset_path="$HF_REPO_ID" \
  -T split=train \
  -T input_field=question \
  -T target_field=answer_text \
  -T filter_field="subject;format" \
  -T filter_value="biology;multiple_choice,matching" \
  -T grader_model="openai/$GRADER_MODEL_ID" \
  --batch false \
  -M responses_api=false
```

**Example 3: Evaluating questions with Image Descriptions (Physics)**

```bash
uv run inspect eval src/evals/tasks.py \
  --model "openai/$MODEL_ID" \
  --max-connections 10 \
  -T dataset_path="$HF_REPO_ID" \
  -T split=train \
  -T input_field=question \
  -T target_field=answer_text \
  -T filter_field="subject;has_image_description" \
  -T filter_value="physics;true" \
  -T grader_model="openai/$GRADER_MODEL_ID" \
  --batch false \
  -M responses_api=false
```
</details>

## Local Data & Repository Contents

For researchers working locally or using the source repository, the data is available in several formats with additional internal metadata for traceability.

**data/**: *(Currently withheld for the double-blind review process)* Contains the raw structured files (JSON for questions, MD for answers) categorized by subject and year.

**notebooks/**: Jupyter notebooks used for data exploration, testing, and pipeline prototyping.

**results/**: *(Currently withheld for the double-blind review process)* Destination folder for the consolidated Excel master files and comparison reports.

**src/**: Core Python modules and processing scripts (e.g., build_dataset.py).

## Getting Started (Developer)

**Prerequisites**

* Environment: Python 3.10+ (Recommended: use uv for fast dependency management).
* API Tokens: Create a .env file in the root directory with your Hugging Face credentials to enable uploading:

```python
HF_TOKEN=your_huggingface_write_token
HF_REPO_ID=your_username/panellinies_exams_dataset
HF_PRIVATE_REPO=False
HF_GATED_REPO=manual
```

## Management Commands

The repository includes a comprehensive management script src/build_dataset.py to handle the data lifecycle via a Command Line Interface (CLI).

1. Data Consolidation (Local)
   Transforms raw JSON/Markdown files into a structured Excel master file located in the results/ folder.

```python
uv run src/build_dataset.py consolidate
```

2. Data Validation & Comparison
   Compares your newly generated dataset against a previous "golden" reference file to ensure no breaking changes were introduced during processing.

```python
uv run src/build_dataset.py compare --reference ../old_file.xlsx
```

3. Pushing to Hugging Face Hub
   Synchronizes the local structured dataset with the Hugging Face Hub.
   To include multimodal assets (embedding the actual pixel data into the HF Parquet files), use the --with-images flag:

```python
uv run src/build_dataset.py push --file results/panellinies_dataset.xlsx --with-images
```

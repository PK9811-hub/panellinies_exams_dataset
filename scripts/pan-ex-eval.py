import os
import ast
import re
import fcntl
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from datasets import load_dataset

# Enable debug logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="Panellinies Human Evaluation", layout="wide")

# Custom CSS for compact Argilla-like dashboard styling
st.markdown("""
<style>
/* Hide default Streamlit top header bar, toolbar, and decoration */
header[data-testid="stHeader"] {
    display: none !important;
}
div[data-testid="stDecoration"] {
    display: none !important;
}
footer {
    display: none !important;
}

/* Hide background communication iframe */
iframe {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
}

/* Centered dashboard layout */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1480px !important;
    margin: 0 auto;
}

/* Field labels with bluish theme */
.field-label {
    font-size: 0.75rem;
    font-weight: 700;
    color: #1976d2;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    margin-bottom: 0.35rem;
}

/* KaTeX formula font styling */
.katex {
    font-size: 1.05em;
}

/* Sticky score panel on the right */
[data-testid="column"]:nth-of-type(2) {
    position: sticky;
    top: 1rem;
}

/* Primary blue buttons */
button[kind="primary"] {
    background-color: #1976d2 !important;
    border-color: #1976d2 !important;
    color: #ffffff !important;
    font-weight: 600 !important;
}
button[kind="primary"]:hover {
    background-color: #1565c0 !important;
    border-color: #1565c0 !important;
}

/* Segmented control / pills with bluish active state */
div[data-testid="stSegmentedControl"] {
    display: flex;
    justify-content: space-between;
    width: 100%;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"] {
    background-color: #1976d2 !important;
    color: #ffffff !important;
    border-color: #1976d2 !important;
}

/* Progress bar in primary blue */
div[data-testid="stProgress"] > div > div > div > div {
    background-color: #1976d2 !important;
}

/* Subtle bluish border for container cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #e0e7ee !important;
}
</style>
""", unsafe_allow_html=True)

# --- Root Directory and Environment ---
def find_project_root() -> Path:
    current = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent
    return current

PROJECT_ROOT = find_project_root()
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file, override=True)

CSV_PATH = PROJECT_ROOT / "scripts" / "human_evaluation_panellinies_science_subset.csv"
HF_REPO = os.getenv("PANELLINIES_DATASET", "ilsp/panellinies-exams-dataset")
APP_PASSWORD = os.getenv("EVAL_APP_PASSWORD")

# --- Simple Authentication (Active only if EVAL_APP_PASSWORD is set in .env) ---
if APP_PASSWORD:
    if "auth_ok" not in st.session_state:
        st.session_state.auth_ok = False

    if not st.session_state.auth_ok:
        with st.container(border=True):
            st.markdown("### 🔒 Evaluation Dashboard Access")
            entered = st.text_input("Enter Password", type="password", key="pass_input")
            if st.button("Unlock", type="primary"):
                if entered == APP_PASSWORD:
                    st.session_state.auth_ok = True
                    st.rerun()
                else:
                    st.error("Incorrect password.")
        st.stop()

# --- Chemistry & LaTeX Normalization (from pass-or-fail-argilla-eval.py) ---
def fix_inline_math(s: str) -> str:
    s = re.sub(r'(?<=[(\[{"\'])\$', ' $', s)
    s = re.sub(r'\$(?=[)\]}"\';])', '$ ', s)
    return s

def clean_science_latex(text: str) -> str:
    if not text or pd.isna(text):
        return ""
    text = str(text).strip()
    if text.lower() == "nan":
        return ""

    text = fix_inline_math(text)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = text.replace('$$', '$')
    text = text.replace(r'\(', '$').replace(r'\)', '$')
    text = text.replace(r'\[', '$').replace(r'\]', '$')

    bugs = {
        "pK_{a}$)": "$pK_{a}$)", "pK_a$)": "$pK_a$)", r"\mathrm{M}$)": r"$\mathrm{M}$)",
        "Y_{2}$)": "$Y_{2}$)", "Y_{1}$ και Y_{2}$)": "$Y_{1}$ και $Y_{2}$)",
        "K_a = 10^{-5} $)": "$K_a = 10^{-5}$)", "aa$)": "aa)", "($aa$)": "(aa)",
        "($Aa": "(Aa", "Aa$)": "Aa)", "\\$": "",
        r"\frac{[CH_3COO^-]}{[CH_3COOH]} = 1 $": r"$\frac{[CH_3COO^-]}{[CH_3COOH]} = 1$",
        r"\cdotpmin": r"\cdot \mathrm{min}", r"\cdotp": r"\cdot",
        "'$": "'", "$-": "-", "-$": "-",
        "< = >": r" \rightleftharpoons ", "<=>": r" \rightleftharpoons ", "⇌": r" \rightleftharpoons ",
        " - > ": r" \rightarrow ", " -> ": r" \rightarrow ", "→": r" \rightarrow ", "−": "-",
        "H_{3}$O+": "H_{3}O+", r"H\Delta$": r"H\Delta", r"\Delta^-$": r"\Delta^-",
        "SO_{3}$": "SO_{3}", "NO_{2}$": "NO_{2}", "SO_{2}$": "SO_{2}", "HO-$": "HO-",
        "C_{6}H_{5}-$": "C_{6}H_{5}-", "NO_{2}$ -": "NO_{2}-"
    }
    for bad, good in bugs.items():
        text = text.replace(bad, good)

    def fix_chem(content):
        content = content.replace('$', '')
        content = re.sub(r'(?<=[A-Za-z)\]])(\d+)', r'_{\1}', content)
        content = re.sub(r'([+-]+)$', r'^{\1}', content)
        return f"\\mathrm{{{content}}}"

    # Replace \ce{...} from chemistry exams
    text = re.sub(r'\\ce\s*{([^}]+)}', lambda m: fix_chem(m.group(1)), text)
    text = re.sub(r'\\ce\s*([A-Za-z0-9_+\-\^]+)', lambda m: fix_chem(m.group(1)), text)

    text = re.sub(r'\$+', '$', text)

    parts = text.split('$')
    if len(parts) % 2 == 0:
        parts.append("")

    for i in range(1, len(parts), 2):
        parts[i] = parts[i].replace('\n', ' ').replace('\r', '')

    text = '$'.join(parts)

    new_parts = text.split('$')
    for i in range(0, len(new_parts), 2):
        new_parts[i] = re.sub(r'(\\mathrm\s*{[^{}]*})', r'$\1$', new_parts[i])
        new_parts[i] = re.sub(r'(\\d?frac\s*{[^{}]*}\s*{[^{}]*})', r'$\1$', new_parts[i])
        new_parts[i] = re.sub(r'(\\rightleftharpoons)', r'$\1$', new_parts[i])
        new_parts[i] = re.sub(r'(\\rightarrow)', r'$\1$', new_parts[i])
        new_parts[i] = re.sub(r'(\\Rightarrow)', r'$\1$', new_parts[i])
        new_parts[i] = re.sub(r'\b(X\^[aA])\b', r'$\1$', new_parts[i])

    text = '$'.join(new_parts)

    text = re.sub(r'\$+', '$', text)
    text = text.replace('$$', '$')
    if text.count('$') % 2 != 0:
        text += '$'

    text = text.replace('\r', '')
    return text

def get_clean_str(val):
    if val is None or pd.isna(val):
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s

# --- Helpers ---
@st.dialog("Figure Preview", width="large")
def show_large_figure(img, caption_text):
    st.image(img, width="stretch", caption=caption_text)

def parse_dict_field(val):
    s = get_clean_str(val)
    if not s:
        return None
    if s.startswith("{") and s.endswith("}"):
        try:
            return ast.literal_eval(s)
        except Exception:
            return None
    return None

def unpack_field(val, preferred_key="reference"):
    parsed = parse_dict_field(val)
    if isinstance(parsed, dict):
        for k in [preferred_key, "answer", "text", "final_answer", "solution"]:
            if k in parsed:
                return get_clean_str(parsed[k])
        if len(parsed) == 1:
            return get_clean_str(next(iter(parsed.values())))
    return get_clean_str(val)

@st.cache_resource(show_spinner="Loading Hugging Face dataset resources...")
def get_hf_index():
    try:
        hf_dataset_dict = load_dataset(HF_REPO)
        index = {}
        for split in hf_dataset_dict.keys():
            for item in hf_dataset_dict[split]:
                index[item["id"]] = item
        return index
    except Exception:
        return {}

hf_index = get_hf_index()

# --- Data Loading, Synchronization & Persistence ---
def load_data(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col, default in [
        ("Human_Grade", None),
        ("Explanation", ""),
        ("Status", "unseen"),
        ("Edited_Question", ""),
        ("Edited_Reference", ""),
        ("Edited_Model_Answer", ""),
    ]:
        if col not in df.columns:
            df[col] = default
        else:
            if col in ["Explanation", "Edited_Question", "Edited_Reference", "Edited_Model_Answer"]:
                df[col] = df[col].fillna("")

    # Order by Question_ID with model responses shuffled deterministically within each question
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.sort_values(by="Question_ID", kind="mergesort").reset_index(drop=True)
    return df

def sync_from_disk(session_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """
    Safely merges completed evaluations and LaTeX edits from disk into the session DataFrame
    without changing row order or disturbing active session widgets.
    """
    if not csv_path.exists():
        return session_df
    try:
        disk_df = pd.read_csv(csv_path, encoding="utf-8-sig")
        if "Question_ID" not in disk_df.columns or "Model" not in disk_df.columns:
            return session_df

        disk_map = {}
        for _, r in disk_df.iterrows():
            k = (str(r.get("Question_ID", "")), str(r.get("Model", "")))
            disk_map[k] = r

        cols_to_sync = ["Status", "Human_Grade", "Explanation", "Edited_Question", "Edited_Reference", "Edited_Model_Answer"]
        for idx, row in session_df.iterrows():
            k = (str(row.get("Question_ID", "")), str(row.get("Model", "")))
            if k in disk_map:
                dr = disk_map[k]
                if dr.get("Status") == "seen":
                    for c in cols_to_sync:
                        if c in dr and pd.notna(dr[c]):
                            session_df.at[idx, c] = dr[c]
                else:
                    for c in ["Edited_Question", "Edited_Reference", "Edited_Model_Answer"]:
                        if c in dr and pd.notna(dr[c]) and str(dr[c]).strip():
                            session_df.at[idx, c] = dr[c]
        return session_df
    except Exception as e:
        logging.warning(f"Could not sync from disk: {e}")
        return session_df

def update_row_on_disk(
    csv_path: Path,
    qid: str,
    model_name: str,
    grade: float | None = None,
    explanation: str | None = None,
    status: str | None = None,
    edited_q: str | None = None,
    edited_ref: str | None = None,
    edited_model: str | None = None,
) -> pd.DataFrame:
    """
    Reloads the latest CSV from disk and updates ONLY the specified row / question
    under an exclusive file lock, then writes back to disk.
    This guarantees that concurrent evaluators never overwrite each other's submissions.
    """
    lock_path = csv_path.with_suffix(".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            disk_df = pd.read_csv(csv_path, encoding="utf-8-sig")
            for col, default in [
                ("Human_Grade", None),
                ("Explanation", ""),
                ("Status", "unseen"),
                ("Edited_Question", ""),
                ("Edited_Reference", ""),
                ("Edited_Model_Answer", ""),
            ]:
                if col not in disk_df.columns:
                    disk_df[col] = default
                elif col in ["Explanation", "Edited_Question", "Edited_Reference", "Edited_Model_Answer"]:
                    disk_df[col] = disk_df[col].fillna("")

            # Find matching record on disk by Question_ID and Model
            row_mask = (disk_df["Question_ID"].astype(str) == str(qid)) & (
                disk_df["Model"].astype(str) == str(model_name)
            )

            if row_mask.any():
                if grade is not None:
                    disk_df.loc[row_mask, "Human_Grade"] = grade
                if explanation is not None:
                    disk_df.loc[row_mask, "Explanation"] = explanation
                if status is not None:
                    disk_df.loc[row_mask, "Status"] = status
                if edited_model is not None:
                    disk_df.loc[row_mask, "Edited_Model_Answer"] = edited_model

            # Question-level LaTeX corrections apply to all rows for this Question_ID
            q_mask = disk_df["Question_ID"].astype(str) == str(qid)
            if q_mask.any():
                if edited_q is not None:
                    disk_df.loc[q_mask, "Edited_Question"] = edited_q
                if edited_ref is not None:
                    disk_df.loc[q_mask, "Edited_Reference"] = edited_ref

            disk_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            return disk_df
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

if not CSV_PATH.exists():
    st.error(f"CSV file not found at: {CSV_PATH}")
    st.stop()

if "df" not in st.session_state:
    st.session_state.df = load_data(CSV_PATH)
else:
    st.session_state.df = sync_from_disk(st.session_state.df, CSV_PATH)

df = st.session_state.df

if "nav_count" not in st.session_state:
    st.session_state.nav_count = 0

def reset_expanders():
    st.session_state.nav_count = st.session_state.get("nav_count", 0) + 1

def get_dirty_fields(check_row_idx):
    if check_row_idx not in df.index:
        return []
    r = df.loc[check_row_idx]
    q_id = str(r["Question_ID"])
    hf_i = hf_index.get(q_id, {})

    input_dict = parse_dict_field(r.get("Input_Question"))
    c_q = get_clean_str(input_dict.get("question", "") if input_dict else r.get("Input_Question"))
    c_ref = unpack_field(r.get("Reference_Target"), preferred_key="reference")
    c_model = unpack_field(r.get("Model_Answer"), preferred_key="answer")

    saved_q = get_clean_str(r.get("Edited_Question")) or clean_science_latex(hf_i.get("question") or c_q)
    saved_ref = get_clean_str(r.get("Edited_Reference")) or clean_science_latex(hf_i.get("answer_text") or c_ref)
    saved_model = get_clean_str(r.get("Edited_Model_Answer")) or clean_science_latex(c_model)

    saved_grade = r.get("Human_Grade")
    try:
        saved_grade_val = float(saved_grade) if pd.notna(saved_grade) and saved_grade is not None else None
    except Exception:
        saved_grade_val = None
    saved_exp = get_clean_str(r.get("Explanation"))

    dirty = []
    if st.session_state.get("allow_latex_edit", False):
        if f"edit_q_{check_row_idx}" in st.session_state:
            curr_q = st.session_state[f"edit_q_{check_row_idx}"]
            if curr_q is not None and get_clean_str(curr_q) != get_clean_str(saved_q):
                dirty.append("Question LaTeX")

        if f"edit_ref_{check_row_idx}" in st.session_state:
            curr_ref = st.session_state[f"edit_ref_{check_row_idx}"]
            if curr_ref is not None and get_clean_str(curr_ref) != get_clean_str(saved_ref):
                dirty.append("Reference LaTeX")

        if f"edit_model_{check_row_idx}" in st.session_state:
            curr_model = st.session_state[f"edit_model_{check_row_idx}"]
            if curr_model is not None and get_clean_str(curr_model) != get_clean_str(saved_model):
                dirty.append("Model Answer LaTeX")

    if f"seg_grade_{check_row_idx}" in st.session_state:
        curr_grade = st.session_state[f"seg_grade_{check_row_idx}"]
        if curr_grade != saved_grade_val:
            dirty.append("Grade")

    if f"text_exp_{check_row_idx}" in st.session_state:
        curr_exp = st.session_state[f"text_exp_{check_row_idx}"]
        if curr_exp is not None and get_clean_str(curr_exp) != get_clean_str(saved_exp):
            dirty.append("Explanation")

    return dirty

@st.dialog("⚠️ Unsaved Changes", width="medium")
def confirm_leave_dialog(target_idx, dirty_fields):
    st.warning(f"You have unsaved changes in: **{', '.join(dirty_fields)}**.")
    st.markdown("If you leave now without saving or submitting, your edits will be discarded.")
    col_d, col_s = st.columns([1, 1])
    with col_d:
        if st.button("Discard & Leave", type="primary", width="stretch"):
            cur = st.session_state.active_row_idx
            for k in [f"edit_q_{cur}", f"edit_ref_{cur}", f"edit_model_{cur}", f"seg_grade_{cur}", f"text_exp_{cur}"]:
                st.session_state.pop(k, None)
            st.session_state.active_row_idx = target_idx
            st.session_state.last_row_idx = target_idx
            reset_expanders()
            st.rerun()
    with col_s:
        if st.button("Stay on Page", width="stretch"):
            st.session_state.nav_count += 1
            st.rerun()

def try_navigate(target_idx, reset_input_key=False):
    if target_idx == st.session_state.active_row_idx:
        return
    dirty = get_dirty_fields(st.session_state.active_row_idx)
    if dirty:
        if reset_input_key:
            st.session_state.nav_count += 1
        confirm_leave_dialog(target_idx, dirty)
    else:
        st.session_state.active_row_idx = target_idx
        st.session_state.last_row_idx = target_idx
        reset_expanders()
        st.rerun()

# --- Top Navigation & Filter Bar ---
with st.container(border=True):
    col_filters, col_nav = st.columns([5.3, 4.7], vertical_alignment="center")

    with col_filters:
        f1, f2, f3, f4 = st.columns([1.0, 1.2, 1.6, 1.5], vertical_alignment="center")
        with f1:
            status_filter = st.selectbox(
                "Status",
                ["All", "Pending Only", "Submitted Only"],
                index=0,
                label_visibility="collapsed"
            )
        with f2:
            subjects = ["All Subjects"] + sorted([str(s) for s in df["Subject"].dropna().unique()])
            selected_subject = st.selectbox(
                "Subject",
                subjects,
                index=0,
                label_visibility="collapsed"
            )
        with f3:
            search_query = st.text_input(
                "Search",
                placeholder="🔍 Search...",
                label_visibility="collapsed"
            )
        with f4:
            allow_latex_edit = st.toggle("✏️ Edit LaTeX", value=False, key="allow_latex_edit")

    # Filtering logic
    filtered_df = df.copy()
    if status_filter == "Pending Only":
        filtered_df = filtered_df[filtered_df["Status"] != "seen"]
    elif status_filter == "Submitted Only":
        filtered_df = filtered_df[filtered_df["Status"] == "seen"]

    if selected_subject != "All Subjects":
        filtered_df = filtered_df[filtered_df["Subject"] == selected_subject]

    if search_query.strip():
        q = search_query.strip().lower()
        mask = (
            filtered_df["Question_ID"].astype(str).str.lower().str.contains(q, regex=False, na=False)
            | filtered_df["Input_Question"].astype(str).str.lower().str.contains(q, regex=False, na=False)
            | filtered_df["Reference_Target"].astype(str).str.lower().str.contains(q, regex=False, na=False)
            | filtered_df["Model_Answer"].astype(str).str.lower().str.contains(q, regex=False, na=False)
        )
        filtered_df = filtered_df[mask]

    if filtered_df.empty:
        st.warning("No records match the current filters.")
        st.stop()

    filtered_row_indices = filtered_df.index.tolist()

    if "nav_count" not in st.session_state:
        st.session_state.nav_count = 0

    # Track active record by unique row index
    if "active_row_idx" not in st.session_state or st.session_state.active_row_idx not in filtered_row_indices:
        st.session_state.active_row_idx = filtered_row_indices[0]

    if "last_row_idx" not in st.session_state:
        st.session_state.last_row_idx = st.session_state.active_row_idx
    elif st.session_state.active_row_idx != st.session_state.last_row_idx:
        reset_expanders()
        st.session_state.last_row_idx = st.session_state.active_row_idx

    curr_pos = filtered_row_indices.index(st.session_state.active_row_idx)

    with col_nav:
        n_prev, n_pos, n_of, n_next, n_jump = st.columns([0.85, 1.1, 0.95, 0.85, 2.2], vertical_alignment="center")
        with n_prev:
            if st.button("◀ Prev", disabled=(curr_pos <= 0), width="stretch"):
                try_navigate(filtered_row_indices[curr_pos - 1])
        with n_pos:
            target_pos = st.number_input(
                "Order",
                min_value=1,
                max_value=len(filtered_row_indices),
                value=curr_pos + 1,
                step=1,
                label_visibility="collapsed",
                key=f"pos_input_{st.session_state.nav_count}"
            )
            if target_pos != curr_pos + 1:
                try_navigate(filtered_row_indices[target_pos - 1], reset_input_key=True)
        with n_of:
            st.markdown(
                f"<div style='font-size: 0.9rem; font-weight: 600; color: #546e7a; white-space: nowrap;'>/ {len(filtered_row_indices)}</div>",
                unsafe_allow_html=True
            )
        with n_next:
            if st.button("Next ▶", disabled=(curr_pos >= len(filtered_row_indices) - 1), width="stretch"):
                try_navigate(filtered_row_indices[curr_pos + 1])
        with n_jump:
            # Build unique question IDs preserving order
            unique_qids = []
            seen_qids = set()
            for idx in filtered_row_indices:
                q = str(df.loc[idx, "Question_ID"])
                if q not in seen_qids:
                    seen_qids.add(q)
                    unique_qids.append(q)

            curr_qid = str(df.loc[st.session_state.active_row_idx, "Question_ID"])
            curr_qid_idx = unique_qids.index(curr_qid) if curr_qid in unique_qids else 0

            def jump_label(q_id):
                q_rows = df[df["Question_ID"] == q_id]
                all_seen = (q_rows["Status"] == "seen").all() if not q_rows.empty else False
                badge = "🟢" if all_seen else "⚪"
                return f"{badge} {q_id}"

            selected_qid = st.selectbox(
                "Jump to",
                options=unique_qids,
                index=curr_qid_idx,
                format_func=jump_label,
                label_visibility="collapsed",
                key=f"jump_qid_{st.session_state.nav_count}"
            )
            if selected_qid != curr_qid:
                matching = [i for i in filtered_row_indices if str(df.loc[i, "Question_ID"]) == selected_qid]
                if matching:
                    target_idx = next((i for i in matching if df.loc[i, "Status"] != "seen"), matching[0])
                    try_navigate(target_idx, reset_input_key=True)

# Current item data
row_idx = st.session_state.active_row_idx
row = df.loc[row_idx]
qid = str(row["Question_ID"])
model_name = str(row["Model"])
hf_item = hf_index.get(qid, {})

input_q_dict = parse_dict_field(row.get("Input_Question"))
csv_question = get_clean_str(input_q_dict.get("question", "") if input_q_dict else row.get("Input_Question"))
csv_context = get_clean_str(input_q_dict.get("context", "") if input_q_dict else "")
csv_ref = unpack_field(row.get("Reference_Target"), preferred_key="reference")
csv_model = unpack_field(row.get("Model_Answer"), preferred_key="answer")

# Clean science LaTeX / chemistry \ce in all fields
base_question = clean_science_latex(hf_item.get("question") or csv_question)
base_context = clean_science_latex(hf_item.get("input") or csv_context)
base_ref = clean_science_latex(hf_item.get("answer_text") or csv_ref)
base_model = clean_science_latex(csv_model)

edited_q = get_clean_str(row.get("Edited_Question"))
edited_ref = get_clean_str(row.get("Edited_Reference"))
edited_model = get_clean_str(row.get("Edited_Model_Answer"))

active_question = edited_q if edited_q else base_question
active_ref = edited_ref if edited_ref else base_ref
active_model = edited_model if edited_model else base_model

# --- Main Columns: Content (68%), Scoring Panel (32%) ---
col_content, col_score = st.columns([68, 32], gap="large")

with col_content:
    # 1. Question ID Card
    with st.container(border=True):
        st.markdown("<div class='field-label'>Question ID</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='display: flex; align-items: center; gap: 0.8rem;'>"
            f"<span style='font-size: 1.05rem; font-weight: 600; font-family: monospace; color: #1565c0; background: #e3f2fd; padding: 2px 8px; border-radius: 4px; border: 1px solid #bbdefb;'>{qid}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    # 2. Reading Context (if present)
    if base_context and str(base_context).strip():
        with st.container(border=True):
            st.markdown("<div class='field-label'>Context / Reading Passage</div>", unsafe_allow_html=True)
            st.markdown(str(base_context))

    # 3. Question Images (compact thumbnail with click to enlarge)
    images = hf_item.get("images", [])
    if images:
        with st.container(border=True):
            st.markdown("<div class='field-label'>Images</div>", unsafe_allow_html=True)
            img_cols = st.columns(min(len(images), 4))
            for i, img in enumerate(images):
                with img_cols[i % len(img_cols)]:
                    st.image(img, width=190, caption=f"Figure {i+1}")
                    with st.popover(f"🔍 Enlarge Fig. {i+1}", width="stretch"):
                        st.image(img, width="stretch", caption=f"{qid} — Figure {i+1}")

            with st.expander("🖼️ View All Figures Full-Width", expanded=False, key=f"exp_figs_{st.session_state.nav_count}"):
                for i, img in enumerate(images):
                    st.image(img, width="stretch", caption=f"Figure {i+1}")

    # 4. Question Card (Rendered + LaTeX correction expander with Live Preview)
    with st.container(border=True):
        st.markdown("<div class='field-label'>Question</div>", unsafe_allow_html=True)
        st.markdown(active_question)
        if allow_latex_edit:
            with st.expander("✏️ Correct Question LaTeX", expanded=False, key=f"exp_q_{st.session_state.nav_count}"):
                new_q = st.text_area("Question LaTeX", value=active_question, height=120, key=f"edit_q_{row_idx}")
                if new_q != active_question:
                    active_question = new_q
                st.markdown("<div style='font-size: 0.75rem; font-weight: 700; color: #1976d2; margin-top: 0.6rem; text-transform: uppercase;'>Live KaTeX Preview:</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(new_q)
                if st.button("💾 Save Question LaTeX Only", key=f"save_btn_q_{row_idx}", type="secondary"):
                    df.loc[df["Question_ID"] == qid, "Edited_Question"] = new_q
                    try:
                        update_row_on_disk(CSV_PATH, qid=qid, model_name=model_name, edited_q=new_q)
                        st.session_state.df = df
                        st.toast(f"Saved Question LaTeX for {qid} (all responses)!", icon="✅")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error saving to CSV: {err}")

    # 5. Reference Answer Card (Rendered + LaTeX correction expander with Live Preview)
    with st.container(border=True):
        st.markdown("<div class='field-label'>Reference Answer</div>", unsafe_allow_html=True)
        st.markdown(active_ref)
        if allow_latex_edit:
            with st.expander("✏️ Correct Reference LaTeX", expanded=False, key=f"exp_ref_{st.session_state.nav_count}"):
                new_ref = st.text_area("Reference Answer LaTeX", value=active_ref, height=130, key=f"edit_ref_{row_idx}")
                if new_ref != active_ref:
                    active_ref = new_ref
                st.markdown("<div style='font-size: 0.75rem; font-weight: 700; color: #1976d2; margin-top: 0.6rem; text-transform: uppercase;'>Live KaTeX Preview:</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(new_ref)
                if st.button("💾 Save Reference LaTeX Only", key=f"save_btn_ref_{row_idx}", type="secondary"):
                    df.loc[df["Question_ID"] == qid, "Edited_Reference"] = new_ref
                    try:
                        update_row_on_disk(CSV_PATH, qid=qid, model_name=model_name, edited_ref=new_ref)
                        st.session_state.df = df
                        st.toast(f"Saved Reference LaTeX for {qid} (all responses)!", icon="✅")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error saving to CSV: {err}")

    # 6. Model Answer Card (Rendered + LaTeX correction expander; Model Name is strictly hidden)
    with st.container(border=True):
        st.markdown("<div class='field-label'>Model Answer</div>", unsafe_allow_html=True)
        st.markdown(active_model)
        if allow_latex_edit:
            with st.expander("✏️ Correct Model Answer LaTeX", expanded=False, key=f"exp_model_{st.session_state.nav_count}"):
                new_model = st.text_area("Model Answer LaTeX", value=active_model, height=150, key=f"edit_model_{row_idx}")
                if new_model != active_model:
                    active_model = new_model
                st.markdown("<div style='font-size: 0.75rem; font-weight: 700; color: #1976d2; margin-top: 0.6rem; text-transform: uppercase;'>Live KaTeX Preview:</div>", unsafe_allow_html=True)
                with st.container(border=True):
                    st.markdown(new_model)
                if st.button("💾 Save Model Answer LaTeX Only", key=f"save_btn_model_{row_idx}", type="secondary"):
                    df.at[row_idx, "Edited_Model_Answer"] = new_model
                    try:
                        update_row_on_disk(CSV_PATH, qid=qid, model_name=model_name, edited_model=new_model)
                        st.session_state.df = df
                        st.toast(f"Saved Model Answer LaTeX for {qid}!", icon="✅")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Error saving to CSV: {err}")

with col_score:
    with st.container(border=True):
        is_seen = row.get("Status") == "seen"
        status_badge = (
            "<span style='font-size: 0.8rem; font-weight: 600; padding: 2px 8px; border-radius: 4px; background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9;'>● Submitted</span>"
            if is_seen else
            "<span style='font-size: 0.8rem; font-weight: 600; padding: 2px 8px; border-radius: 4px; background: #fff3e0; color: #e65100; border: 1px solid #ffe0b2;'>● Pending</span>"
        )
        st.markdown(
            f"<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;'>"
            f"<span style='font-size: 1.05rem; font-weight: 700;'>Grade <span style='color: #d32f2f;'>*</span></span>"
            f"{status_badge}"
            f"</div>",
            unsafe_allow_html=True
        )

        grade_options = [0.0, 0.25, 0.50, 0.75, 1.00]
        cur_grade = row.get("Human_Grade")
        try:
            cur_grade_val = float(cur_grade) if pd.notna(cur_grade) and cur_grade is not None else None
        except Exception:
            cur_grade_val = None

        # Clean segmented pill buttons: None is selected by default for pending items
        selected_grade = st.segmented_control(
            "Grade Selection",
            options=grade_options,
            default=cur_grade_val,
            format_func=lambda x: f"{x:.2f}",
            key=f"seg_grade_{row_idx}",
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-top: 1.2rem; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.3rem;'>Explanation</div>", unsafe_allow_html=True)
        explanation_val = st.text_area(
            "Explanation",
            value=get_clean_str(row.get("Explanation")),
            height=130,
            placeholder="Write explanation / justification in Greek...",
            key=f"text_exp_{row_idx}",
            label_visibility="collapsed"
        )

        st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
        col_discard, col_submit = st.columns([1, 1])

        with col_discard:
            if st.button("Discard", width="stretch"):
                cur = row_idx
                for k in [f"edit_q_{cur}", f"edit_ref_{cur}", f"edit_model_{cur}", f"seg_grade_{cur}", f"text_exp_{cur}"]:
                    st.session_state.pop(k, None)
                reset_expanders()
                st.toast("Changes discarded.", icon="↩️")
                st.rerun()

        with col_submit:
            submit_clicked = st.button("Submit", type="primary", width="stretch")

        if submit_clicked:
            if selected_grade is None:
                st.error("Please select a Grade (0.00 – 1.00) before submitting.")
            else:
                # Update DataFrame
                df.at[row_idx, "Human_Grade"] = selected_grade
                df.at[row_idx, "Explanation"] = explanation_val
                df.at[row_idx, "Status"] = "seen"
                df.loc[df["Question_ID"] == qid, "Edited_Question"] = active_question
                df.loc[df["Question_ID"] == qid, "Edited_Reference"] = active_ref
                df.at[row_idx, "Edited_Model_Answer"] = active_model

                try:
                    update_row_on_disk(
                        CSV_PATH,
                        qid=qid,
                        model_name=model_name,
                        grade=selected_grade,
                        explanation=explanation_val,
                        status="seen",
                        edited_q=active_question,
                        edited_ref=active_ref,
                        edited_model=active_model,
                    )
                    st.session_state.df = sync_from_disk(df, CSV_PATH)
                    st.toast(f"Saved {qid} with Grade {selected_grade:.2f}!", icon="✅")

                    cur = row_idx
                    for k in [f"edit_q_{cur}", f"edit_ref_{cur}", f"edit_model_{cur}", f"seg_grade_{cur}", f"text_exp_{cur}"]:
                        st.session_state.pop(k, None)

                    # Advance to next item if available
                    pos = filtered_row_indices.index(row_idx)
                    if pos < len(filtered_row_indices) - 1:
                        next_row = filtered_row_indices[pos + 1]
                        st.session_state.active_row_idx = next_row
                        st.session_state.last_row_idx = next_row
                    reset_expanders()
                    st.rerun()
                except Exception as err:
                    st.error(f"Error saving to CSV: {err}")

        # Progress bar
        seen_total = len(df[df["Status"] == "seen"])
        total_items = len(df)
        pct = (seen_total / total_items * 100) if total_items > 0 else 0
        st.markdown("<hr style='margin: 1.2rem 0 0.8rem 0;' />", unsafe_allow_html=True)
        st.caption(f"**Submitted:** {seen_total} / {total_items} ({pct:.1f}%)")
        st.progress(seen_total / total_items if total_items > 0 else 0)

# Clear any legacy onbeforeunload handler on the window/parent
st.components.v1.html(
    """
    <script>
    try {
        window.onbeforeunload = null;
        if (window.parent) window.parent.onbeforeunload = null;
        if (window.top) window.top.onbeforeunload = null;
    } catch(e) {}
    </script>
    """,
    height=0,
    width=0
)

import shutil
import sys
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CSV_PATH = SCRIPT_DIR / "human_evaluation_panellinies_science_subset.csv"
BACKUP_PATH = SCRIPT_DIR / "human_evaluation_panellinies_science_subset.csv.test_bak"
APP_PATH = SCRIPT_DIR / "pan-ex-eval.py"

def run_test():
    print("=== STARTING HEADLESS INTEGRATION TEST FOR SUBMISSION FLOW ===")
    
    # 1. Create a safe backup of the current CSV
    shutil.copy2(CSV_PATH, BACKUP_PATH)
    print(f"[1/5] Backup created at: {BACKUP_PATH.name}")

    try:
        # 2. Initialize AppTest with mocked load_dataset for instant loading
        print("[2/5] Initializing AppTest from pan-ex-eval.py...")
        from unittest.mock import patch
        with patch("datasets.load_dataset", return_value={}):
            at = AppTest.from_file(str(APP_PATH), default_timeout=30)
            at.run()
        assert not at.exception, f"App launched with exception: {at.exception}"
        print("  -> App successfully loaded headlessly.")

        initial_row_idx = at.session_state.active_row_idx
        print(f"  -> Initial active row index: {initial_row_idx}")

        # 3. Test Validation: Submit without selecting grade
        print("[3/5] Testing validation: Submitting without Grade selection...")
        submit_button = None
        for btn in at.button:
            if btn.label == "Submit":
                submit_button = btn
                break
        assert submit_button is not None, "Submit button not found in AppTest!"
        
        # Click submit with no grade selected
        submit_button.click().run()
        assert not at.exception, f"Exception on empty submit: {at.exception}"
        assert len(at.error) > 0, "Expected st.error when submitting without grade!"
        assert "Please select a Grade" in at.error[0].value, f"Unexpected error message: {at.error[0].value}"
        assert at.session_state.active_row_idx == initial_row_idx, "Row advanced despite missing grade!"
        print("  -> Validation passed: Submission blocked with correct error message.")

        # 4. Test Successful Submission
        print("[4/5] Testing valid submission: Grade=0.75, Explanation='Δοκιμή επιτυχούς υποβολής'...")
        # Find segmented control or set session_state directly for the widget
        grade_key = f"seg_grade_{initial_row_idx}"
        exp_key = f"text_exp_{initial_row_idx}"
        
        at.session_state[grade_key] = 0.75
        at.session_state[exp_key] = "Δοκιμή επιτυχούς υποβολής"
        
        # Click submit
        # Re-find the submit button in the current tree
        submit_button = None
        for btn in at.button:
            if btn.label == "Submit":
                submit_button = btn
                break
        assert submit_button is not None, "Submit button not found!"
        submit_button.click().run()
        assert not at.exception, f"Exception on valid submit: {at.exception}"

        # Check navigation advanced
        new_row_idx = at.session_state.active_row_idx
        assert new_row_idx != initial_row_idx, f"Row index did not advance! ({initial_row_idx} -> {new_row_idx})"
        print(f"  -> Navigation advanced from row {initial_row_idx} to row {new_row_idx}.")

        # Check dirty session state was popped
        assert grade_key not in at.session_state, f"Session state key {grade_key} was not cleared!"
        assert exp_key not in at.session_state, f"Session state key {exp_key} was not cleared!"
        print("  -> Session state widget keys for previous row successfully cleared.")

        # 5. Verify CSV on disk
        print("[5/5] Verifying persistent disk writes in CSV...")
        df_disk = pd.read_csv(CSV_PATH)
        row_disk = df_disk.loc[initial_row_idx]
        
        assert row_disk["Status"] == "seen", f"Status not updated to 'seen'! Value: {row_disk['Status']}"
        assert float(row_disk["Human_Grade"]) == 0.75, f"Human_Grade not 0.75! Value: {row_disk['Human_Grade']}"
        assert row_disk["Explanation"] == "Δοκιμή επιτυχούς υποβολής", f"Explanation mismatch! Value: {row_disk['Explanation']}"
        print("  -> Disk verification passed:")
        print(f"     - Status: {row_disk['Status']}")
        print(f"     - Human_Grade: {row_disk['Human_Grade']}")
        print(f"     - Explanation: {row_disk['Explanation']}")

        print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! The submission workflow works as expected.")

    finally:
        # Restore original CSV
        if BACKUP_PATH.exists():
            shutil.copy2(BACKUP_PATH, CSV_PATH)
            BACKUP_PATH.unlink()
            print("  -> Cleaned up: Production CSV restored from backup.")

if __name__ == "__main__":
    run_test()

"""
Copy ilsp/benchmark_name to a personal Hugging Face account to test whether the
Dataset Viewer works there (personal repos are stored in the US region. The ilsp
original is in the EU region).

The copy is never publicly downloadable: the repo is created empty, set to
gated=manual, the gate is verified, and only then are the files uploaded. It is never
private, for two reasons: the viewer is disabled for private datasets of free, non-PRO
accounts (which would make the test meaningless), and uploads into private repos count
against the 100 GB private-storage quota of free accounts (a full quota makes the
commit fail with "403 Private repository storage limit reached").

Needs a token with WRITE access to your own account. Your current read-only token
cannot create repositories. Put the write token in .env as HF_WRITE_TOKEN.

The entire repository is copied, including all config/subset directories and splits.


Usage:
    python hf_viewer_test_copy_new.py
        Dry run: checks only, writes nothing.

    python hf_viewer_test_copy_new.py --execute
        Creates the personal copy and watches the Dataset Viewer.

    python hf_viewer_test_copy_new.py --delete
        Deletes the personal test copy.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()

SOURCE_REPO = "ilsp/panellinies-exams-dataset"
TARGET_NAME = "panellinies-exams-dataset"
VIEWER_API = "https://datasets-server.huggingface.co"

TEST_BANNER = (
    "> **Temporary copy for a Dataset Viewer test. Do not use or cite this repository.**\n"
    f"> The official benchmark is [{SOURCE_REPO}](https://huggingface.co/datasets/{SOURCE_REPO}).\n\n"
)


def get_token():
    token = os.getenv("HF_WRITE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        sys.exit("No token found: set HF_WRITE_TOKEN (or HF_TOKEN) in .env.")
    return token


def check_token(api, token):
    """Return username and abort if token cannot write."""
    info = api.whoami(token=token)
    access = (info.get("auth") or {}).get("accessToken") or {}
    role = access.get("role")

    print(f"Token belongs to '{info['name']}', role = {role!r}")

    if role == "read":
        sys.exit(
            "\nThis token is READ-only, so it cannot create a repository.\n"
            "Create a token with write access at https://huggingface.co/settings/tokens\n"
            "and put it in .env as HF_WRITE_TOKEN=hf_..."
        )

    return info["name"]


def viewer_status(repo_id, token):
    params = urllib.parse.urlencode({"dataset": repo_id})
    req = urllib.request.Request(
        f"{VIEWER_API}/is-valid?{params}",
        headers={"Authorization": f"Bearer {token}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {
            "error": f"HTTP {e.code}",
            "x_error_code": e.headers.get("X-Error-Code", ""),
        }


def get_splits(repo_id, token):
    params = urllib.parse.urlencode({"dataset": repo_id})
    req = urllib.request.Request(
        f"{VIEWER_API}/splits?{params}",
        headers={"Authorization": f"Bearer {token}"},
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read()).get("splits", [])
    except urllib.error.HTTPError as e:
        code = e.headers.get("X-Error-Code", "")
        try:
            msg = json.loads(e.read()).get("error", "")
        except Exception:
            msg = ""

        print(f"Could not retrieve splits for {repo_id}: HTTP {e.code} {code}: {msg}")
        return []


def first_rows_status(repo_id, token):
    splits = get_splits(repo_id, token)

    if not splits:
        return ["No configs/splits found yet."]

    results = []

    for item in splits:
        config = item["config"]
        split = item["split"]

        params = urllib.parse.urlencode({
            "dataset": repo_id,
            "config": config,
            "split": split,
        })

        req = urllib.request.Request(
            f"{VIEWER_API}/first-rows?{params}",
            headers={"Authorization": f"Bearer {token}"},
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                rows = json.loads(r.read()).get("rows", [])
                results.append(f"{config}/{split}: OK, {len(rows)} preview rows served")
        except urllib.error.HTTPError as e:
            code = e.headers.get("X-Error-Code", "")
            try:
                msg = json.loads(e.read()).get("error", "")
            except Exception:
                msg = ""

            results.append(f"{config}/{split}: HTTP {e.code} {code}: {msg}")
        except Exception as e:
            results.append(f"{config}/{split}: ERROR: {e}")

    return results


def rows_status(repo_id, token, length=1):
    splits = get_splits(repo_id, token)

    if not splits:
        return ["No configs/splits found yet."]

    results = []

    for item in splits:
        config = item["config"]
        split = item["split"]

        params = urllib.parse.urlencode({
            "dataset": repo_id,
            "config": config,
            "split": split,
            "offset": 0,
            "length": length,
        })

        req = urllib.request.Request(
            f"{VIEWER_API}/rows?{params}",
            headers={"Authorization": f"Bearer {token}"},
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                json.loads(r.read())
                results.append(f"{config}/{split}: OK")
        except urllib.error.HTTPError as e:
            code = e.headers.get("X-Error-Code", "")
            try:
                msg = json.loads(e.read()).get("error", "")
            except Exception:
                msg = ""

            results.append(f"{config}/{split}: HTTP {e.code} {code}: {msg}")
        except Exception as e:
            results.append(f"{config}/{split}: ERROR: {e}")

    return results


def create_copy(api, token, target_repo):
    print(f"Creating {target_repo} (empty) ...")
    api.create_repo(target_repo, repo_type="dataset", private=False, exist_ok=True, token=token)

    print("Setting gated=manual ...")
    api.update_repo_settings(target_repo, repo_type="dataset", gated="manual", token=token)
    api.update_repo_settings(target_repo, repo_type="dataset", private=False, token=token)

    info = api.dataset_info(target_repo, token=token)

    if info.private or info.gated != "manual":
        sys.exit(
            f"Gate check failed (private={info.private}, gated={info.gated!r}); nothing was uploaded."
        )

    print("Repo is public and gated (manual approval).")

    with tempfile.TemporaryDirectory() as tmp:
        print(f"Downloading complete repository {SOURCE_REPO} ...")

        local = Path(
            snapshot_download(
                SOURCE_REPO,
                repo_type="dataset",
                local_dir=tmp,
                token=token,
            )
        )

        readme = local / "README.md"

        if readme.exists():
            text = readme.read_text(encoding="utf-8")

            if text.startswith("---"):
                try:
                    end = text.index("---", 3) + 3
                    text = text[:end] + "\n\n" + TEST_BANNER + text[end:].lstrip("\n")
                except ValueError:
                    text = TEST_BANNER + text
            else:
                text = TEST_BANNER + text

            readme.write_text(text, encoding="utf-8")
        else:
            readme.write_text(TEST_BANNER, encoding="utf-8")

        print("Uploading complete dataset repository ...")

        api.upload_folder(
            folder_path=str(local),
            repo_id=target_repo,
            repo_type="dataset",
            commit_message="Temporary copy for a Dataset Viewer test",
            token=token,
        )

    print(f"Done: https://huggingface.co/datasets/{target_repo}")


def watch_viewer(target_repo, token, minutes=10):
    print(f"\nWatching the Dataset Viewer for up to {minutes} minutes ...")

    deadline = time.time() + minutes * 60

    while time.time() < deadline:
        status = viewer_status(target_repo, token)
        print(f"  {time.strftime('%H:%M:%S')}  {status}")

        if status.get("viewer") is True:
            print("\nViewer reports viewer=True.")

            print("\nDiscovered configs/splits:")
            splits = get_splits(target_repo, token)

            for item in splits:
                print(f"  {item['config']} / {item['split']}")

            print("\nTesting /first-rows for every config/split:")
            first_results = first_rows_status(target_repo, token)

            for result in first_results:
                print(f"  {result}")

            print("\nTesting /rows for every config/split:")
            rows_results = rows_status(target_repo, token)

            for result in rows_results:
                print(f"  {result}")

            all_rows_ok = all(": OK" in result for result in rows_results)

            if all_rows_ok:
                print("\nRESULT: the viewer WORKS on the personal-account copy.")
                print(
                    "=> This strongly suggests the original ILSP/EU-hosted repository "
                    "is responsible for the Viewer problem rather than the dataset data."
                )
            else:
                print("\nRESULT: viewer=True, but /rows still fails for at least one config/split.")
                print(
                    "=> The problem is probably not specific to the original "
                    "ILSP/EU-hosted repository."
                )

            return

        time.sleep(30)

    print("\nRESULT: the viewer is not ready yet.")

    print("\nDiscovered configs/splits:")
    splits = get_splits(target_repo, token)

    if splits:
        for item in splits:
            print(f"  {item['config']} / {item['split']}")
    else:
        print("  None discovered yet.")

    print("\n/first-rows status:")
    for result in first_rows_status(target_repo, token):
        print(f"  {result}")

    print("\n/rows status:")
    for result in rows_status(target_repo, token):
        print(f"  {result}")

    print("\nCheck again later with:")
    print("  python hf_viewer_test_copy.py")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true", help="create the copy (default: dry run)")
    group.add_argument("--delete", action="store_true", help="delete the test copy")
    args = parser.parse_args()

    token = get_token()
    api = HfApi()

    if not (args.execute or args.delete):
        info = api.whoami(token=token)
        target_repo = f"{info['name']}/{TARGET_NAME}"
        access = (info.get("auth") or {}).get("accessToken") or {}
        role = access.get("role")

        print("DRY RUN (nothing is written)")
        print(f"  token user / role : {info['name']} / {role}")

        files = api.list_repo_files(SOURCE_REPO, repo_type="dataset", token=token)

        print(f"\nSource repository contains {len(files)} files:")
        for file in files:
            print(f"  {file}")

        print(f"\n  source viewer : {viewer_status(SOURCE_REPO, token)}")

        print("\nSource configs/splits:")
        source_splits = get_splits(SOURCE_REPO, token)

        if source_splits:
            for item in source_splits:
                print(f"  {item['config']} / {item['split']}")
        else:
            print("  None discovered.")

        print(f"\nTarget:\n  {target_repo}")

        if api.repo_exists(target_repo, repo_type="dataset", token=token):
            print(f"\n  target viewer : {viewer_status(target_repo, token)}")

            print("\n  target first-rows:")
            for result in first_rows_status(target_repo, token):
                print(f"    {result}")

            print("\n  target rows:")
            for result in rows_status(target_repo, token):
                print(f"    {result}")
        else:
            print("  target exists : no")

        if role == "read":
            print("\nNOTE: this token is read-only; --execute requires HF_WRITE_TOKEN.")

        return

    user = check_token(api, token)
    target_repo = f"{user}/{TARGET_NAME}"

    if args.delete:
        answer = input(
            f"Permanently delete https://huggingface.co/datasets/{target_repo} ? [y/N] "
        )

        if answer.strip().lower() == "y":
            api.delete_repo(
                target_repo,
                repo_type="dataset",
                token=token,
                missing_ok=True,
            )
            print("Deleted.")
        else:
            print("Cancelled.")

        return

    create_copy(api, token, target_repo)
    watch_viewer(target_repo, token)


if __name__ == "__main__":
    main()
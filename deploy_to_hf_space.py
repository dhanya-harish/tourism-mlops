#!/usr/bin/env python3
"""
Deploy files to a Hugging Face Space.

Usage (matches your workflow step):
  python deploy_to_hf_space.py \
    --space-id dhani10/tourism-app \
    --sdk streamlit \
    --hardware cpu-basic \
    --path . \
    --set-secret HF_TOKEN=$HF_TOKEN \
    --set-var HF_MODEL_REPO=dhani10/tourism-model \
    --set-var MODEL_FILE=model/best_model.joblib
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import login, HfApi, create_repo, upload_folder
from huggingface_hub.utils import HfHubHTTPError

def parse_kv_list(items):
    """Convert ["KEY=VALUE", ...] into dict."""
    out = {}
    if not items:
        return out
    for s in items:
        if "=" not in s:
            raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {s}")
        k, v = s.split("=", 1)
        out[k] = v
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--space-id", required=True, help="e.g. user/space-name")
    p.add_argument("--path", default=".", help="Folder to upload (default: .)")
    p.add_argument("--sdk", choices=["streamlit", "docker"], default="streamlit",
                   help="Space runtime SDK (default: streamlit)")
    p.add_argument("--hardware", default="cpu-basic",
                   help="Space hardware (e.g. cpu-basic, t4-small)")
    p.add_argument("--set-secret", nargs="*", help="Secrets as KEY=VALUE")
    p.add_argument("--set-var", nargs="*", help="Variables as KEY=VALUE")
    args = p.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN env var is required (GitHub Secret).")

    login(token=hf_token)
    api = HfApi(token=hf_token)

    space_id = args.space_id
    root = Path(args.path).resolve()

    # Create or ensure Space exists (avoid passing sdk at creation on older hubs)
    try:
        exists = api.repo_exists(space_id, repo_type="space")
    except Exception:
        exists = False

    if not exists:
        try:
            create_repo(
                repo_id=space_id,
                repo_type="space",
                private=True,
                exist_ok=False,
                token=hf_token,
            )
            print(f"[info] Created Space: {space_id}")
        except HfHubHTTPError as e:
            if "already exists" in str(e).lower():
                print("[info] Space already exists.")
            else:
                print(f"[warn] create_repo: {e}")

    # Try to set runtime (newer client supports update_space_runtime)
    try:
        api.update_space_runtime(
            repo_id=space_id,
            sdk=args.sdk,
            hardware=args.hardware,
            token=hf_token,
        )
        print(f"[info] Runtime set: sdk={args.sdk}, hw={args.hardware}")
    except Exception as e:
        print(f"[info] update_space_runtime not available or failed: {e}")

    # Upload files
    if not root.exists():
        raise SystemExit(f"Path not found: {root}")

    # If using Streamlit SDK, make sure only needed files are present
    # (app.py and requirements.txt at minimum). This avoids accidental Docker use.
    if args.sdk == "streamlit":
        must_have = ["app.py", "requirements.txt"]
        missing = [f for f in must_have if not (root / f).exists()]
        if missing:
            raise SystemExit(f"Missing required files for Streamlit SDK: {missing}")
        # Warn if a Dockerfile is present (it would force Docker runtime)
        if (root / "Dockerfile").exists():
            print("[warn] Dockerfile found. Delete it to ensure Streamlit SDK is used.")

    upload_folder(
        folder_path=str(root),
        repo_id=space_id,
        repo_type="space",
        token=hf_token,
    )
    print("[info] Files uploaded.")

    # Add secrets and variables if provided
    for k, v in parse_kv_list(args.set_secret).items():
        try:
            api.add_space_secret(repo_id=space_id, key=k, value=v)
            print(f"[info] Secret set: {k}")
        except Exception as e:
            print(f"[warn] add_space_secret({k}): {e}")

    for k, v in parse_kv_list(args.set_var).items():
        try:
            api.add_space_variable(repo_id=space_id, key=k, value=v)
            print(f"[info] Variable set: {k}={v}")
        except Exception as e:
            print(f"[warn] add_space_variable({k}): {e}")

    # Restart Space (best effort)
    try:
        api.restart_space(repo_id=space_id, token=hf_token)
        print("[info] Space restart requested.")
    except Exception as e:
        print(f"[info] restart_space not available or failed: {e}")

    print(f"✅ Deployed to https://huggingface.co/spaces/{space_id}")

if __name__ == "__main__":
    main()

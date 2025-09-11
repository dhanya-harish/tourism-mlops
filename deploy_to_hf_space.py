#!/usr/bin/env python3
"""
Deploy (or update) a Hugging Face Space.

Usage examples
--------------
# Streamlit SDK Space (app.py + requirements.txt)
python deploy_to_hf_space.py --space-id dhani10/tourism-app --sdk streamlit \
  --path . --hardware cpu-basic \
  --set-secret HF_TOKEN=$HF_TOKEN \
  --set-var HF_MODEL_REPO=dhani10/tourism-model --set-var MODEL_FILE=model/best_model.joblib

# Docker Space (Dockerfile + app.py + requirements.txt)
python deploy_to_hf_space.py --space-id dhani10/tourism-app --sdk docker \
  --path . --hardware cpu-basic \
  --set-secret HF_TOKEN=$HF_TOKEN \
  --set-var HF_MODEL_REPO=dhani10/tourism-model --set-var MODEL_FILE=model/best_model.joblib
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import HfApi, login, create_repo, upload_file

SUPPORTED_SDKS = {"streamlit", "gradio", "docker"}
DEFAULT_INCLUDE = {
    "streamlit": ["app.py", "requirements.txt"],
    "gradio":    ["app.py", "requirements.txt"],
    "docker":    ["Dockerfile", "app.py", "requirements.txt"],
}

def parse_kv(items: List[str]) -> List[Tuple[str, str]]:
    """Parse KEY=VALUE pairs."""
    kv_list = []
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE for: {item}")
        k, v = item.split("=", 1)
        kv_list.append((k.strip(), v.strip()))
    return kv_list

def main():
    p = argparse.ArgumentParser(description="Deploy to Hugging Face Space")
    p.add_argument("--space-id", required=True, help="e.g., 'dhani10/tourism-app'")
    p.add_argument("--sdk", choices=SUPPORTED_SDKS, default="streamlit",
                   help="Space runtime: streamlit | gradio | docker")
    p.add_argument("--hardware", default="cpu-basic",
                   help="Space hardware: cpu-basic (default), t4-small, a10g-large, etc.")
    p.add_argument("--path", default=".", help="Folder to upload files from")
    p.add_argument("--include", nargs="*", default=None,
                   help="Specific files to upload (defaults depend on runtime)")
    p.add_argument("--private", action="store_true", help="Create Space as private")
    p.add_argument("--token", default=os.getenv("HF_TOKEN"),
                   help="HF token (or set env HF_TOKEN)")
    p.add_argument("--set-secret", nargs="*", default=[],
                   help="Space secrets to set: KEY=VALUE (e.g., HF_TOKEN=...)")
    p.add_argument("--set-var", nargs="*", default=[],
                   help="Space variables to set: KEY=VALUE (visible env vars)")

    args = p.parse_args()

    if not args.token:
        print("ERROR: Provide a Hugging Face token via --token or env HF_TOKEN", file=sys.stderr)
        sys.exit(1)

    space_id = args.space_id
    sdk = args.sdk
    path = Path(args.path)
    include = args.include or DEFAULT_INCLUDE[sdk]
    include_paths = [path / f for f in include]

    # Sanity checks
    missing = [str(p) for p in include_paths if not p.exists()]
    if missing:
        print("ERROR: Missing required files:\n  - " + "\n  - ".join(missing), file=sys.stderr)
        sys.exit(1)

    # Login
    print("🔐 Logging in to Hugging Face Hub…")
    login(args.token)

    api = HfApi()

    # Create Space (or ensure it exists) with proper SDK + hardware
    print(f"🚀 Creating/ensuring Space '{space_id}' (sdk={sdk}, hardware={args.hardware})…")
    # create_repo supports setting Space meta on creation time
    create_repo(
        repo_id=space_id,
        repo_type="space",
        exist_ok=True,
        private=args.private,
        space_sdk=sdk,
        space_hardware=args.hardware,
        token=args.token,
    )

    # If you need to change SDK/hardware after creation, you can re-call create_repo with same params.

    # Set secrets (encrypted, not visible to users)
    for key, value in parse_kv(args.set_secret):
        print(f"🔑 Setting secret: {key}")
        api.add_space_secret(repo_id=space_id, key=key, value=value, token=args.token)

    # Set variables (plain env vars visible in UI)
    for key, value in parse_kv(args.set_var):
        print(f"🧩 Setting variable: {key}={value}")
        api.add_space_variable(repo_id=space_id, key=key, value=value, token=args.token)

    # Upload files
    print("📤 Uploading files:")
    for pth in include_paths:
        rel = pth.relative_to(path).as_posix()
        print(f"  - {rel}")
        upload_file(
            path_or_fileobj=str(pth),
            path_in_repo=rel,
            repo_id=space_id,
            repo_type="space",
            token=args.token,
        )

    # Optional: upload everything under `assets/` (if exists)
    assets_dir = path / "assets"
    if assets_dir.exists() and assets_dir.is_dir():
        for fp in assets_dir.rglob("*"):
            if fp.is_file():
                rel = fp.relative_to(path).as_posix()
                print(f"  - {rel}")
                upload_file(
                    path_or_fileobj=str(fp),
                    path_in_repo=rel,
                    repo_id=space_id,
                    repo_type="space",
                    token=args.token,
                )

    # Restart Space to pick new commit
    print("🔁 Restarting Space…")
    try:
        api.restart_space(repo_id=space_id, token=args.token)
    except Exception as e:
        # Not fatal: Space will rebuild automatically on commit anyway
        print(f"(info) restart_space not available or failed: {e}")

    app_url = f"https://huggingface.co/spaces/{space_id}"
    print(f"✅ Done. Space URL: {app_url}")

if __name__ == "__main__":
    main()

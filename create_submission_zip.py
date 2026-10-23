"""Create clean submission ZIP adhering to Page 14 instructions:
- Includes: README.md, .env.example, requirements.txt, run.py, run.sh, src/, report.pdf, sample_outputs/
- Excludes: .env, API keys, virtual environments (venv), dataset copy, __pycache__
"""
import os
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def package_submission(zip_name="team_name_voiceops.zip"):
    output_zip = PROJECT_ROOT / zip_name
    print(f"Creating submission package: {output_zip.name}...")

    # Files and folders to include
    include_files = [
        "README.md",
        ".env.example",
        "requirements.txt",
        "run.py",
        "run.sh",
        "report.pdf",
        "DEMO_GUIDE.md"
    ]
    include_dirs = ["src", "sample_outputs"]

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        # Add root files
        for f_name in include_files:
            f_path = PROJECT_ROOT / f_name
            if f_path.exists():
                z.write(f_path, arcname=f_name)
                print(f"  + Added: {f_name}")
            else:
                print(f"  - Warning: Missing {f_name}")

        # Add directories
        for d_name in include_dirs:
            d_path = PROJECT_ROOT / d_name
            if d_path.exists():
                for root, dirs, files in os.walk(d_path):
                    # Exclude __pycache__ and hidden files
                    dirs[:] = [d for d in dirs if d != "__pycache__" and not d.startswith(".")]
                    for file in files:
                        if file.endswith(".pyc") or file.startswith("."):
                            continue
                        full_p = Path(root) / file
                        rel_p = full_p.relative_to(PROJECT_ROOT)
                        z.write(full_p, arcname=str(rel_p))
                        print(f"  + Added: {rel_p}")

    print(f"\nSuccessfully generated {output_zip.name} ({output_zip.stat().st_size} bytes).")
    print("Secrets, .env, virtual environments, and dataset copies were strictly excluded.")

if __name__ == "__main__":
    package_submission()

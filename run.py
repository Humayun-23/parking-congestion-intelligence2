import os
import sys
import subprocess
from pathlib import Path
import venv

def main():
    root = Path(__file__).resolve().parent
    venv_dir = root / ".venv"

    # Create virtual environment if it doesn't exist
    if not venv_dir.exists():
        print(f"==> creating virtual environment (.venv) with {sys.executable}")
        venv.create(venv_dir, with_pip=True)

    # Determine executable paths
    if os.name == 'nt':
        venv_python = venv_dir / "Scripts" / "python.exe"
    else:
        venv_python = venv_dir / "bin" / "python"

    # Install dependencies
    print("==> installing pinned dependencies")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
    subprocess.run([str(venv_python), "-m", "pip", "install", "--quiet", "-r", str(root / "requirements.txt")], check=True)

    # Determine mode
    mode = sys.argv[1] if len(sys.argv) > 1 else "app"

    # Run pipeline if requested
    if mode in ("all", "pipeline"):
        print("==> running analysis pipeline (01 -> 06)")
        subprocess.run([str(venv_python), str(root / "src" / "run_pipeline.py")], check=True)

    # Run app if requested
    if mode in ("all", "app"):
        print("==> launching Streamlit app at http://localhost:8501")
        try:
            subprocess.run([str(venv_python), "-m", "streamlit", "run", str(root / "app" / "streamlit_app.py")])
        except KeyboardInterrupt:
            print("\nExiting Streamlit app.")

if __name__ == "__main__":
    main()

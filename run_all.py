"""Run data generation, train/test split, and model training."""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run(cmd):
    print(f"\n>>> {cmd}")
    r = subprocess.run([sys.executable] + cmd.split(), cwd=os.getcwd())
    if r.returncode != 0:
        print("FAILED:", cmd)
        sys.exit(r.returncode)

run("data_generator.py")
run("train_model.py")
print("\nDone. Start dashboard with: streamlit run app.py")

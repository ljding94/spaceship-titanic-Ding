#!/usr/bin/env python3
"""
Spaceship Titanic — Auto-Improve Script
Weekly Claude Code-driven model improvement.
Best LB: 0.799 | Target: 0.81+
"""
import subprocess
import sys
import os

REPO = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(REPO, '.venv', 'bin', 'python')
BEST_LB = 0.799

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO)
    if result.returncode != 0:
        print(f"ERROR: {result.stderr}")
        return False
    return True

def main():
    print("=" * 50)
    print("SPACESHIP TITANIC — Auto Improve (Claude Edition)")
    print("=" * 50)

    # Delegate to Claude Code for the actual ML work
    prompt = """Improve the Spaceship Titanic model in this directory (~/Work/ljding94-spaceship-titanic-Ding/).

Current best: LB 0.799 (baseline_xgb.py)
Our ensemble (build_model.py) got LB 0.796 despite CV 0.807 — overfitting issue.

Please try a DIFFERENT approach:
1. CatBoost (handles categoricals natively, may reduce overfitting)
2. OR simpler features + stronger regularization  
3. OR Optuna hyperparameter tuning for the baseline features
4. OR a neural network (simple MLP)

Requirements:
- Must beat LB 0.799 to push/submit
- Use .venv Python: ~/Work/ljding94-spaceship-titanic-Ding/.venv/bin/python
- Run 5-fold CV
- If improved, save submission.csv, git add/commit/push
- Update README.md LB tracker with results
- Print what approach was tried and why it worked or didn't
"""
    cmd = f'cd {REPO} && claude -p {repr(prompt)} --max-turns 20 --output-format json --dangerously-skip-permissions'
    
    print("\nLaunching Claude Code...\n")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print("\nDone! Check git log for any improvements.")
    return 0

if __name__ == '__main__':
    sys.exit(main())

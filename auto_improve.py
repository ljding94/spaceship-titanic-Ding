#!/usr/bin/env python3
"""
Spaceship Titanic — Auto-Improve Script
Claude Code multi-round model improvement while you sleep.
Best LB: 0.799 | Target: 0.81+
"""
import subprocess
import sys
import os
import json

REPO = os.path.dirname(os.path.abspath(__file__))
PYTHON = os.path.join(REPO, '.venv', 'bin', 'python')
BEST_LB = 0.799
ROUNDS = 3  # number of Claude invocations per night

# CRITICAL: Set KAGGLE_API_TOKEN from kaggle.json OR ~/.zshrc so the SDK can auth
# (the env var must be set to the actual token, not empty)
_KAGGLE_JSON = os.path.expanduser('~/.kaggle/kaggle.json')
if os.path.exists(_KAGGLE_JSON):
    with open(_KAGGLE_JSON) as _f:
        _creds = json.load(_f)
    os.environ['KAGGLE_API_TOKEN'] = _creds.get('key', '')
else:
    # Fallback: parse KAGGLE_API_TOKEN from ~/.zshrc
    _ZSHRC = os.path.expanduser('~/.zshrc')
    if os.path.exists(_ZSHRC):
        with open(_ZSHRC) as _f:
            for line in _f:
                if line.strip().startswith('export KAGGLE_API_TOKEN='):
                    _token = line.strip().split('=', 1)[1].strip().strip('"').strip("'")
                    os.environ['KAGGLE_API_TOKEN'] = _token
                    break


def run_claude_round(round_num, task):
    print(f"\n{'='*50}")
    print(f"  ROUND {round_num}")
    print(f"{'='*50}")
    cmd = (
        f"cd {REPO} && claude -p {repr(task)} "
        f"--max-turns 20 --output-format json --dangerously-skip-permissions"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:500])
    try:
        parsed = json.loads(result.stdout)
        cost = parsed.get("total_cost_usd", "N/A")
        turns = parsed.get("num_turns", "N/A")
        print(f"  → Cost: ${cost}, Turns: {turns}")
    except Exception:
        pass
    return result.returncode == 0


def main():
    print("=" * 50)
    print("SPACESHIP TITANIC — Auto-Improve (Multi-Round)")
    print(f"Target: beat LB {BEST_LB} | Rounds: {ROUNDS}")
    print("=" * 50)

    # Round 1: Try CatBoost approach
    round1_task = """Improve the Spaceship Titanic model in this directory (~Work/ljding94-spaceship-titanic-Ding/).

Current best: LB 0.799 (baseline_xgb.py) — simple XGBoost with basic features.
The ensemble (build_model.py) got CV 0.807 but LB 0.796 — overfitting.

**CRITICAL: You must complete ALL steps below — do not stop at writing the script.**

1. Write/improve the model script (CatBoost or your chosen approach)
2. RUN the script: `.venv/bin/python your_script.py`
3. Verify it produced a submission_*.csv file with realistic True/False distribution
4. Submit to Kaggle: `export PATH="$HOME/Library/Python/3.12/bin:$PATH" && kaggle competitions submit spaceship-titanic -f submission_YOURNAME.csv -m "description"`
5. Check the public LB score returned by Kaggle
6. IF score improved: git add/commit/push AND update README.md LB Tracker table with the new score
7. Report all results (CV, LB score, what was submitted)

Python: ~/Work/ljding94-spaceship-titanic-Ding/.venv/bin/python
Data: train.csv (8693 rows), test.csv (4277 rows)
Kaggle auth: token at ~/.kaggle/kaggle.json (401 = expired — warn if so)"""
    success1 = run_claude_round(1, round1_task)

    # Round 2: Try simpler regularization approach
    round2_task = """Continue improving the Spaceship Titanic model.

Previous result: Check git log for what was tried. Read the current model file.

Try a DIFFERENT approach this round:
1. XGBoost with heavy regularization (min_child_weight high, subsample low)
2. Fewer features — remove the engineered ones, go back to basics
3. Or try a simple Neural Network (MLP) with sklearn MLPClassifier
4. 5-fold CV, compare carefully

**CRITICAL: Complete ALL steps — do not stop at writing the script.**
1. Write/improve the model script
2. RUN the script: `.venv/bin/python your_script.py`
3. Verify submission file with realistic True/False counts
4. Submit: `export PATH="$HOME/Library/Python/3.12/bin:$PATH" && kaggle competitions submit spaceship-titanic -f submission_NAME.csv -m "description"`
5. Check LB score returned
6. IF improved: git add/commit/push AND update README.md LB Tracker
7. Report all results

Python: ~/Work/ljding94-spaceship-titanic-Ding/.venv/bin/python
Kaggle auth: ~/.kaggle/kaggle.json (401 = expired — warn if so)"""
    success2 = run_claude_round(2, round2_task)

    # Round 3: Try Optuna hyperparameter tuning
    round3_task = """Final push on the Spaceship Titanic model.

Try Optuna hyperparameter tuning for XGBoost or LightGBM:
1. Use Optuna to search: learning_rate, max_depth, n_estimators, reg_alpha, reg_lambda
2. 5-fold CV as objective
3. Use the baseline feature set (don't over-engineer)

**CRITICAL: Complete ALL steps — do not stop at writing the script.**
1. Write the Optuna tuning script
2. RUN it: `.venv/bin/python your_script.py`
3. Verify submission_*.csv with realistic True/False counts
4. Submit: `export PATH="$HOME/Library/Python/3.12/bin:$PATH" && kaggle competitions submit spaceship-titanic -f submission_NAME.csv -m "description"`
5. Check LB score returned
6. IF improved: git add/commit/push AND update README.md LB Tracker
7. Report all results (CV, LB, what was submitted)

Python: ~/Work/ljding94-spaceship-titanic-Ding/.venv/bin/python
Kaggle auth: ~/.kaggle/kaggle.json (401 = expired — warn if so)"""
    run_claude_round(3, round3_task)

    print("\n" + "=" * 50)
    print("All rounds complete. Check git log for results.")
    print("=" * 50)
    return 0


if __name__ == '__main__':
    sys.exit(main())

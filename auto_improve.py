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

Try this approach:
1. CatBoost with native categorical handling (no encoding needed)
2. Use only the baseline features (keep it simple to avoid overfitting)
3. 5-fold cross-validation
4. If LB > 0.799: save submission.csv, git add/commit/push, update README.md

Python: ~/Work/ljding94-spaceship-titanic-Ding/.venv/bin/python
Data: train.csv (8693 rows), test.csv (4277 rows)

IMPORTANT: Only push if LB score actually improved."""
    success1 = run_claude_round(1, round1_task)

    # Round 2: Try simpler regularization approach
    round2_task = """Continue improving the Spaceship Titanic model.

Previous result: Check git log for what was tried. Read the current model file.

Try a DIFFERENT approach this round:
1. XGBoost with heavy regularization (min_child_weight high, subsample low)
2. Fewer features — remove the engineered ones, go back to basics
3. Or try a simple Neural Network (MLP) with sklearn MLPClassifier
4. 5-fold CV, compare carefully

If improved (LB > 0.799): save submission.csv, git add/commit/push, update README.
If not improved: try a different approach, still commit the code for reference.

Python: ~/Work/ljding94-spaceship-titanic-Ding/.venv/bin/python"""
    success2 = run_claude_round(2, round2_task)

    # Round 3: Try Optuna hyperparameter tuning
    round3_task = """Final push on the Spaceship Titanic model.

Try Optuna hyperparameter tuning for XGBoost or LightGBM:
1. Use Optuna to search: learning_rate, max_depth, n_estimators, reg_alpha, reg_lambda
2. 5-fold CV as objective
3. Use the baseline feature set (don't over-engineer)

If best trial CV > 0.805: generate submission and push.
Even if not better: commit the Optuna study results for future reference.

Python: ~/Work/ljding94-spaceship-titanic-Ding/.venv/bin/python"""
    run_claude_round(3, round3_task)

    print("\n" + "=" * 50)
    print("All rounds complete. Check git log for results.")
    print("=" * 50)
    return 0


if __name__ == '__main__':
    sys.exit(main())

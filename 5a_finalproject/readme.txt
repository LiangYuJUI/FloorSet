================================================================================
ICCAD 2026 FloorSet Challenge - Final Project Submission
Team: pd26s048 (Yu-Jui Liang, R14946013)
Directory: 5a_finalproject/
================================================================================

OVERVIEW
--------
This submission implements a constraint-aware B*-tree simulated annealing
floorplanner (test_optimizer.py) for the ICCAD 2026 FloorSet Challenge.
The solver handles preplaced/fixed blocks, MIB groups, boundary chains,
and adaptive soft-penalty calibration during search.

DIRECTORY STRUCTURE
-------------------
5a_finalproject/
  readme.txt                 This file
  report.pdf                 Algorithm description and experimental results
  requirements.txt           Python dependencies
  run_evaluation.sh          Helper script to evaluate or re-score
  cost.py, utils.py, ...     FloorSet dataset / evaluation support modules
  iccad2026contest/
    test_optimizer.py        Main solver (MyOptimizer class)
    iccad2026_evaluate.py    Validation evaluator and scoring framework
    plot_results.py          Generate result figures from JSON
    report.tex               LaTeX source for report.pdf
    results/
      test_optimizer_results.json   Saved validation run (100 cases)
      results_summary.png           Result plots
      runtime_vs_blocks.png
      cost_vs_blocks.png

NOTE: There is no separate compile step. The program is Python and runs
directly with python3.

IMPORTANT — FIRST-TIME SETUP
----------------------------
This submission does NOT include a pre-built virtual environment (.venv/).
You must create one locally before running any commands below.

If you see:
  bash: .venv/bin/activate: No such file or directory
run the "First-time setup" block in Section 1 once, then proceed.


1. DEVELOPMENT ENVIRONMENT SETUP
--------------------------------

Requirements:
  - Python 3.10 or newer (tested with Python 3.13)
  - pip
  - Optional: pdflatex (only if you want to rebuild report.pdf from report.tex)

First-time setup (run once from the submission root):

  cd 5a_finalproject
  python3 -m venv .venv
  source .venv/bin/activate        # Linux / macOS
  # .venv\Scripts\activate         # Windows
  pip install -r requirements.txt

After this, .venv/ will exist and "source .venv/bin/activate" will work in
every new terminal session before you run the program.

Detailed steps:

  (a) Change to the submission root:
        cd 5a_finalproject

  (b) Create a virtual environment (skip if .venv/ already exists):
        python3 -m venv .venv

  (c) Activate the virtual environment:
        source .venv/bin/activate        # Linux / macOS
        # .venv\Scripts\activate         # Windows

      Your shell prompt should change to show (.venv). If activation fails,
      go back to step (b) — the venv has not been created yet.

  (d) Install dependencies (once per venv, or after requirements change):
        pip install -r requirements.txt

  (e) Download the validation dataset (required for full evaluation):

      The evaluator expects the validation data at:
        5a_finalproject/LiteTensorDataTest/

      Download from Hugging Face (~15 MB):
        https://huggingface.co/datasets/IntelLabs/FloorSet

      Extract LiteTensorDataTest/ into 5a_finalproject/ so the path is:
        5a_finalproject/LiteTensorDataTest/config0.pt  (etc., 100 cases)

      Alternatively, on first run the dataset loader may prompt to download
      automatically if the data path is set correctly.


2. HOW TO RUN THE PROGRAM
-------------------------

Prerequisites: complete Section 1 (First-time setup) at least once so that
.venv/ exists and dependencies are installed.

All commands below assume:
  cd 5a_finalproject
  source .venv/bin/activate    # create .venv first if this command fails

(A) Quick re-score of saved results (no optimizer re-run, ~5 seconds):

      ./run_evaluation.sh --score

    or manually:
      cd iccad2026contest
      python iccad2026_evaluate.py --score results/test_optimizer_results.json --data-path ..

    Expected output (reference run):
      Total Score: 7.0969
      Tests: 100
      Feasible: 98
      Avg Cost: 5.8267
      Avg Runtime: 288.24s

    Note: Official costs are recomputed from HPWL gap, area gap, and soft
    violations. The raw "cost" field inside the JSON may differ slightly
    from the printed Avg Cost.

(B) Run optimizer on a single test case (debugging):

      ./run_evaluation.sh --test-id 0

    Test IDs are 0-99 (block counts 21-120).

(C) Full validation evaluation (100 cases, several hours):

      ./run_evaluation.sh

    or manually:
      cd iccad2026contest
      python iccad2026_evaluate.py --evaluate test_optimizer.py \
          --data-path .. --output results/test_optimizer_results.json

(D) Generate result figures:

      cd iccad2026contest
      python plot_results.py

    Produces cost_vs_blocks.png, runtime_vs_blocks.png, results_summary.png
    in the iccad2026contest/ directory.

(E) Validate submission format:

      cd iccad2026contest
      python iccad2026_evaluate.py --validate test_optimizer.py


3. MAIN SOURCE FILE
-------------------
  iccad2026contest/test_optimizer.py

  Entry point: class MyOptimizer(FloorplanOptimizer), method solve().
  The evaluator loads this module dynamically via --evaluate test_optimizer.py.


4. REBUILD report.pdf (OPTIONAL)
--------------------------------
  cd iccad2026contest
  pdflatex report.tex
  pdflatex report.tex
  cp report.pdf ../report.pdf

  For Chinese team-name rendering, XeLaTeX with xeCJK is preferred:
  xelatex report.tex  (requires texlive-xetex and Noto CJK fonts)


5. REPRODUCING REPORTED RESULTS
-------------------------------
  The included results/test_optimizer_results.json contains positions and
  metrics from our validation run. To verify without re-running the optimizer:

      ./run_evaluation.sh --score

  To reproduce figures shown in report.pdf:

      cd iccad2026contest && python plot_results.py

  Full re-evaluation requires LiteTensorDataTest/ and run_evaluation.sh
  without --score (expect ~8+ hours depending on hardware).


6. CONTACT
----------
  Yu-Jui Liang (pd26s048, R14946013)

================================================================================

"""
run_all_models.py
=================
Delete existing model artifacts and regenerate them by executing
all 5 training notebooks for each of 3 tickers (ELSA, DEWA, BUMI).

Usage:
    py run_all_models.py
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────
TICKERS = ["ELSA", "DEWA", "BUMI"]

REPO_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = REPO_ROOT / "artifacts" / "notebooks"
MODELS_DIR = REPO_ROOT / "models"

# Notebook definitions
NOTEBOOKS = [
    "model_prophet.ipynb",
    "model_hybrid_prophet_xgboost.ipynb",
    "model_neuralprophet.ipynb",
    "model_nhits.ipynb",
    "model_nbeats.ipynb",
]

# Expected output files per ticker
EXPECTED_SUFFIXES = [
    "_prophet.joblib",
    "_hybrid.joblib",
    "_neuralprophet_meta.joblib",
    "_nhits.joblib",
    "_nhits.darts",
    "_nbeats.joblib",
]


def delete_existing_models():
    """Delete all files in the models/ directory."""
    if not MODELS_DIR.exists():
        print(f"  📁 models/ directory does not exist, creating it...")
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return

    files = list(MODELS_DIR.iterdir())
    if not files:
        print("  📁 models/ directory is already empty.")
        return

    print(f"  🗑️  Deleting {len(files)} files from models/...")
    for f in files:
        try:
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
            print(f"    ❌ Deleted: {f.name}")
        except Exception as e:
            print(f"    ⚠️  Failed to delete {f.name}: {e}")


def is_visualization_cell(source_lines):
    """Check if a cell contains only visualization/plotting code (no export)."""
    source = "".join(source_lines)
    has_plot = any(kw in source for kw in ["plt.show()", "plt.figure", "plt.savefig", "ax1.plot", "ax1.scatter", "fig.add_subplot"])
    has_export = any(kw in source for kw in ["joblib.dump", ".save(", "model.save", "nhits_model.save"])
    # Only skip if it's purely visualization (not an export cell that also plots)
    return has_plot and not has_export


def is_summary_cell(source_lines):
    """Check if a cell is a post-export summary/insights cell."""
    source = "".join(source_lines)
    has_insights = any(kw in source for kw in ["KEY INSIGHTS", "Analysis Completed", "prophet_errors =", "nhits_errors ="])
    has_export = any(kw in source for kw in ["joblib.dump", ".save(", "model.save"])
    return has_insights and not has_export


def patch_notebook(nb_path: Path, ticker: str) -> dict:
    """
    Read a notebook JSON and patch:
    1. The ticker/emiten variable
    2. Wrap visualization/plotting cells in try/except (or replace with pass)
    3. Fix known issues (e.g., matplotlib backend for headless execution)
    Returns the patched notebook dict.
    """
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    cells = nb.get("cells", [])
    nb_name = nb_path.name

    # Add matplotlib backend setting to first code cell
    backend_set = False

    for i, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue

        source_lines = cell.get("source", [])
        new_lines = []
        patched = False

        # Set matplotlib to non-interactive backend (first code cell only)
        if not backend_set:
            import_lines = [
                "import matplotlib\n",
                "matplotlib.use('Agg')  # Non-interactive backend for headless execution\n",
            ]
            # Prepend before existing imports
            new_lines = import_lines + list(source_lines)
            source_lines = new_lines
            new_lines = []
            backend_set = True

        for line in source_lines:
            # Patch emiten = 'XXX'
            if re.match(r"^emiten\s*=\s*['\"]", line):
                new_lines.append(f"emiten = '{ticker}'\n" if line.endswith("\n") else f"emiten = '{ticker}'")
                patched = True
            # Patch hardcoded yf.download('XXXX.JK', ...) for nbeats
            elif "yf.download(" in line and ".JK" in line and "emiten" not in line:
                new_line = re.sub(
                    r"yf\.download\(\s*['\"][A-Z]+\.JK['\"]",
                    f"yf.download('{ticker}.JK'",
                    line,
                )
                new_lines.append(new_line)
                patched = True
            # Patch hardcoded export_path for nbeats
            elif "export_path" in line and "_nbeats.joblib" in line and "emiten" not in line:
                new_line = re.sub(
                    r"['\"][A-Z]+_nbeats\.joblib['\"]",
                    f"'{ticker}_nbeats.joblib'",
                    line,
                )
                new_lines.append(new_line)
                patched = True
            # Replace plt.show() with pass (avoid GUI issues in headless mode)
            elif "plt.show()" in line:
                new_lines.append(line.replace("plt.show()", "plt.close('all')  # headless mode"))
                patched = True
            else:
                new_lines.append(line)

        if patched:
            cell["source"] = new_lines
        else:
            cell["source"] = new_lines if new_lines else source_lines

        # Skip/wrap visualization and summary cells that can crash
        if is_visualization_cell(cell.get("source", [])) or is_summary_cell(cell.get("source", [])):
            original_source = "".join(cell["source"])
            wrapped = [
                "try:\n",
                "    pass  # Visualization skipped in batch mode\n",
                "    " + original_source.replace("\n", "\n    ") + "\n",
                "except Exception as _viz_err:\n",
                "    print(f'⚠️ Visualization skipped: {_viz_err}')\n",
            ]
            cell["source"] = wrapped

    # Fix N-HiTS specific issue: the prediction cell uses past_covariates
    # that don't start early enough. We need to pass the full covariates.
    if "nhits" in nb_name:
        _fix_nhits_prediction_cells(cells)

    # Fix NeuralProphet specific issue: shape mismatch in visualization
    if "neuralprophet" in nb_name:
        _fix_neuralprophet_cells(cells)

    return nb


def _fix_nhits_prediction_cells(cells):
    """Fix N-HiTS prediction cells to use proper covariate alignment."""
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # The prediction cell that calls model.predict with past_covariates
        # For validation, we need to use full covariates instead of just val_cov
        if "val_forecast_s = model.predict" in source and "val_cov_ts_s" in source:
            # Replace val_cov_ts_s with full_cov_ts_s in the val prediction line
            new_source = source.replace(
                "val_forecast_s = model.predict(n=len(val_df_pd), series=train_y_s, past_covariates=val_cov_ts_s)",
                "val_forecast_s = model.predict(n=len(val_df_pd), series=train_y_s, past_covariates=full_cov_ts_s)"
            )
            cell["source"] = [new_source]


def _fix_neuralprophet_cells(cells):
    """Fix NeuralProphet visualization cell shape mismatch."""
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        # Fix the visualization cell that plots train_dates and test_dates
        # The issue is y_test_actual and reconstructed_prices have different lengths
        # from test_dates. Wrap entire visualization in try/except.
        if "train_dates" in source and "reconstructed_prices" in source and "ax1.plot" in source:
            cell["source"] = [
                "try:\n",
                "    " + source.replace("\n", "\n    ") + "\n",
                "except Exception as _e:\n",
                "    print(f'⚠️ Visualization skipped: {_e}')\n",
            ]


def run_notebook(nb_name: str, ticker: str, run_index: int, total: int) -> bool:
    """
    Patch a notebook for the given ticker and execute it via nbconvert.
    Returns True if execution succeeded.
    """
    nb_path = NOTEBOOKS_DIR / nb_name
    if not nb_path.exists():
        print(f"    ⚠️  Notebook not found: {nb_path}")
        return False

    # Patch the notebook
    patched_nb = patch_notebook(nb_path, ticker)

    # Write temporary patched notebook
    tmp_name = f"_tmp_{ticker}_{nb_name}"
    tmp_path = NOTEBOOKS_DIR / tmp_name
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(patched_nb, f, ensure_ascii=False, indent=1)

    print(f"\n  [{run_index}/{total}] 🏃 Running {nb_name} for {ticker}...")
    start = time.time()

    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--ExecutePreprocessor.timeout=1800",
                "--no-input",
                "--output", f"_output_{ticker}_{nb_name}",
                str(tmp_path),
            ],
            cwd=str(NOTEBOOKS_DIR),
            capture_output=True,
            text=True,
            timeout=2400,
        )

        elapsed = time.time() - start

        if result.returncode == 0:
            print(f"    ✅ Success! ({elapsed:.1f}s)")
            success = True
        else:
            print(f"    ❌ Failed! ({elapsed:.1f}s)")
            stderr_lines = result.stderr.strip().split("\n")
            # Print last 40 lines of stderr for debugging
            for line in stderr_lines[-40:]:
                print(f"       {line}")
            success = False

    except subprocess.TimeoutExpired:
        print(f"    ❌ Timeout after 40 minutes!")
        success = False
    except Exception as e:
        print(f"    ❌ Error: {e}")
        success = False
    finally:
        # Cleanup temp files
        for temp_file in NOTEBOOKS_DIR.glob(f"_tmp_{ticker}_{nb_name}*"):
            try:
                temp_file.unlink()
            except Exception:
                pass
        for temp_file in NOTEBOOKS_DIR.glob(f"_output_{ticker}_{nb_name}*"):
            try:
                temp_file.unlink()
            except Exception:
                pass

    return success


def verify_outputs():
    """Verify that all expected output files exist."""
    print("\n" + "=" * 70)
    print("📋 VERIFICATION")
    print("=" * 70)

    all_ok = True
    for ticker in TICKERS:
        for suffix in EXPECTED_SUFFIXES:
            expected = MODELS_DIR / f"{ticker}{suffix}"
            if expected.exists():
                size_kb = expected.stat().st_size / 1024
                print(f"  ✅ {expected.name} ({size_kb:.1f} KB)")
            else:
                print(f"  ❌ MISSING: {expected.name}")
                all_ok = False

    # Also check for .darts.ckpt files
    for ticker in TICKERS:
        ckpt = MODELS_DIR / f"{ticker}_nhits.darts.ckpt"
        if ckpt.exists():
            size_kb = ckpt.stat().st_size / 1024
            print(f"  ✅ {ckpt.name} ({size_kb:.1f} KB)")
        else:
            print(f"  ⚠️  Optional: {ckpt.name} not found")

    return all_ok


def main():
    print("=" * 70)
    print("🚀 MODEL ARTIFACT REGENERATION SCRIPT")
    print(f"   Tickers: {', '.join(TICKERS)}")
    print(f"   Models:  {len(NOTEBOOKS)} notebooks")
    print(f"   Total:   {len(TICKERS) * len(NOTEBOOKS)} training runs")
    print("=" * 70)

    # Step 1: Delete existing models
    print("\n📌 Step 1: Deleting existing model files...")
    delete_existing_models()

    # Step 2: Run all notebooks
    print("\n📌 Step 2: Training all models...")
    total = len(TICKERS) * len(NOTEBOOKS)
    run_index = 0
    results = []

    overall_start = time.time()

    for ticker in TICKERS:
        print(f"\n{'='*70}")
        print(f"📊 TICKER: {ticker}")
        print(f"{'='*70}")

        for nb_name in NOTEBOOKS:
            run_index += 1
            success = run_notebook(nb_name, ticker, run_index, total)
            results.append((ticker, nb_name, success))

    overall_elapsed = time.time() - overall_start

    # Step 3: Summary
    print(f"\n{'='*70}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total time: {overall_elapsed/60:.1f} minutes")
    print()

    successes = sum(1 for _, _, s in results if s)
    failures = sum(1 for _, _, s in results if not s)
    print(f"  ✅ Succeeded: {successes}/{total}")
    print(f"  ❌ Failed:    {failures}/{total}")

    if failures > 0:
        print("\n  Failed runs:")
        for ticker, nb, success in results:
            if not success:
                print(f"    - {ticker} / {nb}")

    # Step 4: Verify outputs
    all_ok = verify_outputs()

    if all_ok:
        print("\n🎉 All model artifacts regenerated successfully!")
    else:
        print("\n⚠️  Some expected files are missing. Check the failed runs above.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

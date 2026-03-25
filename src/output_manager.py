import os
import shutil
from pathlib import Path


ROOT_FILE_TARGETS = {
    "best_model.joblib": "models",
    "lookup_tables.joblib": "models",
    "surrogate_tree.txt": "models",
    "recall_summary.csv": "reports",
    "rules_summary.csv": "reports",
    "rules_combined_coverage.csv": "reports",
    "test_predictions.csv": "reports",
}


PLOT_FILE_TARGETS = {
    "comparison_": "plots/comparison",
    "shap_": "plots/explainability",
    "rules_": "plots/rules",
    "overfit_": "plots/overfitting",
}


RESERVED_TOP_LEVEL_DIRS = {
    "models",
    "reports",
    "plots",
    "results",
}


def _ensure_layout(output_dir: Path) -> None:
    for rel in (
        "models",
        "reports",
        "plots/comparison",
        "plots/explainability",
        "plots/rules",
        "plots/overfitting",
        "plots/models",
    ):
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def _replace_existing(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _move_with_overwrite(src: Path, dst: Path) -> None:
    src_resolved = src.resolve()
    dst_resolved = dst.resolve() if dst.exists() else dst
    if src_resolved == dst_resolved:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    _replace_existing(dst)
    shutil.move(str(src), str(dst))


def _cleanup_empty_dirs(root: Path) -> None:
    for current, dirs, _files in os.walk(root, topdown=False):
        for dirname in dirs:
            full = Path(current) / dirname
            if full.exists() and full.is_dir() and not any(full.iterdir()):
                full.rmdir()


def reset_output_dir(output_dir: str) -> None:
    """Delete current outputs and recreate a clean output directory skeleton."""
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)
    _ensure_layout(root)


def organize_output_dir(output_dir: str) -> None:
    """Move generated artifacts to a clean, stable output structure."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    _ensure_layout(root)

    # Move known root-level artifacts.
    for filename, target_rel in ROOT_FILE_TARGETS.items():
        src = root / filename
        if src.exists():
            _move_with_overwrite(src, root / target_rel / filename)

    # Move root-level plot files by naming convention.
    for file_path in root.glob("*.png"):
        moved = False
        for prefix, target_rel in PLOT_FILE_TARGETS.items():
            if file_path.name.startswith(prefix):
                _move_with_overwrite(file_path, root / target_rel / file_path.name)
                moved = True
                break
        if not moved:
            _move_with_overwrite(file_path, root / "plots" / file_path.name)

    # Move model-specific top-level directories (e.g. LightGBM, XGBoost).
    for child in root.iterdir():
        if not child.is_dir() or child.name in RESERVED_TOP_LEVEL_DIRS:
            continue
        if any(p.suffix.lower() == ".png" for p in child.rglob("*.png")):
            _move_with_overwrite(child, root / "plots" / "models" / child.name)

    # Pull csv artifacts from legacy nested folders into reports.
    for legacy_rel in ("results", "reports"):
        legacy_dir = root / legacy_rel
        if not legacy_dir.exists() or not legacy_dir.is_dir():
            continue
        for csv_file in legacy_dir.rglob("*.csv"):
            _move_with_overwrite(csv_file, root / "reports" / csv_file.name)

    _cleanup_empty_dirs(root)

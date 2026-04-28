import json
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import joblib
import typer
from tqdm import tqdm
from sklearn.experimental import enable_halving_search_cv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    HalvingGridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
from train_dummy_baseline import (
    TARGET_COLUMN,
    _build_preprocessor,
    _load_csv,
    _resolve_text_column,
    _split_columns,
    _compute_scores,
)

app = typer.Typer()

SEARCH_METHODS = {
    "randomized_search",
    "grid_search",
    "halving_grid_search",
    "halving_random_search",
}
MODEL_NAMES = {"svm", "random_forest", "logistic_regression"}

METHOD_LABELS = {
    "randomized_search": "RandomSearch",
    "grid_search": "GridSearch",
    "halving_grid_search": "HalvingGrid",
    "halving_random_search": "HalvingRandom",
}
MODEL_LABELS = {
    "svm": "LinearSVC",
    "random_forest": "RandomForest",
    "logistic_regression": "LogisticReg",
}


def build_param_grid(model_name: str) -> dict:
    if model_name == "svm":
        return {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__max_iter": [1000, 2000],
            "preprocess__text__vectorizer__max_features": [10000, 20000],
            "preprocess__text__vectorizer__ngram_range": [(1, 1), (1, 2)],
        }
    if model_name == "random_forest":
        return {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 20],
            "model__min_samples_split": [2, 5],
            "preprocess__text__vectorizer__max_features": [10000, 20000],
        }
    if model_name == "logistic_regression":
        return {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__solver": ["saga", "lbfgs"],
            "model__max_iter": [500, 1000],
            "preprocess__text__vectorizer__max_features": [10000, 20000],
        }
    raise ValueError(f"Unknown model_name: {model_name}")


def build_estimator(model_name: str, random_state: int):
    if model_name == "svm":
        return LinearSVC(random_state=random_state)
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(random_state=random_state, n_jobs=-1)
    if model_name == "logistic_regression":
        return LogisticRegression(random_state=random_state, n_jobs=-1)
    raise ValueError(f"Unknown model_name: {model_name}")


def build_base_pipeline(
    model_name: str,
    text_column: str,
    numeric_columns: list,
    categorical_columns: list,
    random_state: int,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
) -> Pipeline:
    preprocessor = _build_preprocessor(
        feature_set="all",
        text_column=text_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        vectorizer_name="tfidf",
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        max_features=max_features,
        feature_selection="none",
        dimension_reduction="none",
        select_k=1000,
        svd_components=300,
        random_state=random_state,
    )
    estimator = build_estimator(model_name, random_state)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])


def build_search_cv(
    method: str,
    pipeline: Pipeline,
    param_grid: dict,
    cv: int,
    n_iter: int,
    random_state: int,
    n_jobs: int,
):
    kwargs = dict(scoring="f1_weighted", cv=cv, n_jobs=n_jobs, refit=True, verbose=0)
    if method == "randomized_search":
        return RandomizedSearchCV(
            pipeline, param_grid, n_iter=n_iter, random_state=random_state, **kwargs
        )
    if method == "grid_search":
        return GridSearchCV(pipeline, param_grid, **kwargs)
    if method == "halving_grid_search":
        return HalvingGridSearchCV(
            pipeline,
            param_grid,
            factor=3,
            min_resources=1000,
            random_state=random_state,
            **kwargs,
        )
    if method == "halving_random_search":
        return HalvingRandomSearchCV(
            pipeline,
            param_grid,
            n_candidates=50,
            factor=3,
            min_resources=1000,
            random_state=random_state,
            **kwargs,
        )
    raise ValueError(f"Unknown search method: {method}")


def count_iterations(search) -> dict:
    n_candidates = len(search.cv_results_["params"])
    total_fits = n_candidates * search.n_splits_
    result = {
        "n_candidates": n_candidates,
        "n_splits": int(search.n_splits_),
        "total_fits": total_fits,
    }
    if hasattr(search, "n_candidates_"):
        result["n_candidates_per_round"] = [int(x) for x in search.n_candidates_]
        result["n_resources_per_round"] = [int(x) for x in search.n_resources_]
        result["n_halving_rounds"] = int(search.n_iterations_)
    return result


def _estimate_total_fits(method: str, param_grid: dict, cv: int, n_iter: int) -> int:
    from itertools import product as iproduct

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    if method == "randomized_search":
        return min(n_iter, n_combos) * cv
    if method == "grid_search":
        return n_combos * cv
    # halving: rough estimate — sum of rounds (factor=3)
    rounds, candidates = 0, n_combos if "grid" in method else min(50, n_combos)
    total = 0
    while candidates >= 1:
        total += candidates * cv
        candidates //= 3
        rounds += 1
    return total


def run_search_for_model(
    method: str,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    text_column: str,
    numeric_columns: list,
    categorical_columns: list,
    cv: int,
    n_iter: int,
    random_state: int,
    n_jobs: int,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
) -> dict:
    pipeline = build_base_pipeline(
        model_name,
        text_column,
        numeric_columns,
        categorical_columns,
        random_state,
        max_features,
        ngram_min,
        ngram_max,
    )
    param_grid = build_param_grid(model_name)
    estimated_fits = _estimate_total_fits(method, param_grid, cv, n_iter)

    search = build_search_cv(
        method, pipeline, param_grid, cv, n_iter, random_state, n_jobs
    )

    fit_bar = tqdm(
        total=estimated_fits,
        desc=f"    fits [{METHOD_LABELS.get(method, method)} × {MODEL_LABELS.get(model_name, model_name)}]",
        unit="fit",
        position=2,
        leave=False,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} fits [{elapsed}<{remaining}, {rate_fmt}]",
    )

    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    runtime = time.perf_counter() - t0

    iterations = count_iterations(search)
    fit_bar.n = iterations["total_fits"]
    fit_bar.total = iterations["total_fits"]
    fit_bar.refresh()
    fit_bar.close()

    tqdm.write(
        f"  ✓ {METHOD_LABELS.get(method, method)} × {MODEL_LABELS.get(model_name, model_name)}"
        f"  f1={search.best_score_:.4f}  fits={iterations['total_fits']}  {runtime:.0f}s"
    )

    return {
        "best_estimator": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "runtime_seconds": float(runtime),
        "iteration_counts": iterations,
    }


def run_all_searches(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    text_column: str,
    numeric_columns: list,
    categorical_columns: list,
    search_methods: list,
    model_names: list,
    cv: int,
    n_iter: int,
    random_state: int,
    n_jobs: int,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
    mlflow_enabled: bool,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
) -> dict:
    if mlflow_enabled:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(mlflow_experiment_name)

    total_runs = len(search_methods) * len(model_names)
    results: dict[str, dict] = {}
    outer_bar = tqdm(search_methods, desc="Search methods", unit="method", position=0)
    for method in outer_bar:
        outer_bar.set_postfix(method=METHOD_LABELS.get(method, method))
        results[method] = {}
        inner_bar = tqdm(
            model_names,
            desc=f"  Models [{METHOD_LABELS.get(method, method)}]",
            unit="model",
            position=1,
            leave=False,
        )
        for model_name in inner_bar:
            inner_bar.set_postfix(model=MODEL_LABELS.get(model_name, model_name))
            run_result = run_search_for_model(
                method=method,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                text_column=text_column,
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                cv=cv,
                n_iter=n_iter,
                random_state=random_state,
                n_jobs=n_jobs,
                max_features=max_features,
                ngram_min=ngram_min,
                ngram_max=ngram_max,
            )
            results[method][model_name] = run_result
            inner_bar.set_postfix(
                model=MODEL_LABELS.get(model_name, model_name),
                f1=f"{run_result['best_score']:.3f}",
                done=True,
            )

            if mlflow_enabled:
                with mlflow.start_run(run_name=f"{method}__{model_name}"):
                    mlflow.set_tags({"search_method": method, "model_name": model_name})
                    mlflow.log_param("search_method", method)
                    mlflow.log_param("model_name", model_name)
                    for k, v in run_result["best_params"].items():
                        mlflow.log_param(f"best_{k}", str(v))
                    mlflow.log_metric("best_cv_f1_weighted", run_result["best_score"])
                    mlflow.log_metric("runtime_seconds", run_result["runtime_seconds"])
                    mlflow.log_metric(
                        "total_fits", run_result["iteration_counts"]["total_fits"]
                    )
                    try:
                        mlflow.sklearn.log_model(
                            run_result["best_estimator"], artifact_path="model"
                        )
                    except Exception as e:
                        print(f"[tune] mlflow log_model failed: {e}", flush=True)

    return results


def save_best_models(
    results: dict, output_dir: Path, model_names: list, search_methods: list
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for method in search_methods:
        method_results = results[method]
        best_model_name = max(
            model_names, key=lambda m: method_results[m]["best_score"]
        )
        best_pipeline = method_results[best_model_name]["best_estimator"]
        out_path = output_dir / f"{method}_best.pkl"
        joblib.dump(best_pipeline, out_path)
        print(
            f"[tune] saved {out_path} (best model: {best_model_name}, score: {method_results[best_model_name]['best_score']:.4f})",
            flush=True,
        )

    for model_name in model_names:
        best_method = max(
            search_methods, key=lambda m: results[m][model_name]["best_score"]
        )
        best_pipeline = results[best_method][model_name]["best_estimator"]
        out_path = output_dir / f"{model_name}_best.pkl"
        joblib.dump(best_pipeline, out_path)
        print(
            f"[tune] saved {out_path} (best method: {best_method}, score: {results[best_method][model_name]['best_score']:.4f})",
            flush=True,
        )


def plot_comparison(
    results: dict, output_path: Path, search_methods: list, model_names: list
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    method_colors = {
        m: c
        for m, c in zip(search_methods, ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    }
    model_markers = {m: mk for m, mk in zip(model_names, ["o", "s", "^"])}
    method_labels = [METHOD_LABELS.get(m, m) for m in search_methods]
    model_labels = [MODEL_LABELS.get(m, m) for m in model_names]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Lab 6: Hyperparameter Tuning Comparison", fontsize=14, fontweight="bold"
    )

    x = np.arange(len(model_names))
    bar_width = 0.2

    # [0,0] F1-weighted scores
    ax = axes[0, 0]
    for i, method in enumerate(search_methods):
        scores = [results[method][m]["best_score"] for m in model_names]
        bars = ax.bar(
            x + i * bar_width,
            scores,
            bar_width,
            label=METHOD_LABELS.get(method, method),
            color=method_colors[method],
        )
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{score:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xlabel("Model")
    ax.set_ylabel("Best CV F1-Weighted")
    ax.set_title("Best CV F1-Weighted Score by Model and Search Method")
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(model_labels)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # [0,1] Runtime
    ax = axes[0, 1]
    for i, method in enumerate(search_methods):
        runtimes = [results[method][m]["runtime_seconds"] for m in model_names]
        bars = ax.bar(
            x + i * bar_width,
            runtimes,
            bar_width,
            label=METHOD_LABELS.get(method, method),
            color=method_colors[method],
        )
        for bar, rt in zip(bars, runtimes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{rt:.0f}s",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xlabel("Model")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Hyperparameter Search Runtime (seconds)")
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels(model_labels)
    ax.legend(fontsize=8)

    # [1,0] Scatter: runtime vs F1
    ax = axes[1, 0]
    for method in search_methods:
        for model_name in model_names:
            r = results[method][model_name]
            ax.scatter(
                r["runtime_seconds"],
                r["best_score"],
                color=method_colors[method],
                marker=model_markers[model_name],
                s=100,
                zorder=3,
            )
            ax.annotate(
                f"{METHOD_LABELS.get(method, method)}\n{MODEL_LABELS.get(model_name, model_name)}",
                (r["runtime_seconds"], r["best_score"]),
                fontsize=6,
                textcoords="offset points",
                xytext=(4, 4),
            )

    from matplotlib.lines import Line2D

    legend_method = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=method_colors[m],
            label=METHOD_LABELS.get(m, m),
            linestyle="None",
        )
        for m in search_methods
    ]
    legend_model = [
        Line2D(
            [0],
            [0],
            marker=model_markers[m],
            color="gray",
            label=MODEL_LABELS.get(m, m),
            linestyle="None",
        )
        for m in model_names
    ]
    ax.legend(handles=legend_method + legend_model, fontsize=7, loc="lower right")
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Best CV F1-Weighted")
    ax.set_title("Quality vs Runtime Tradeoff")

    # [1,1] Total fits
    ax = axes[1, 1]
    combo_labels = []
    combo_fits = []
    combo_colors = []
    for method in search_methods:
        for model_name in model_names:
            combo_labels.append(
                f"{METHOD_LABELS.get(method, method)}\n{MODEL_LABELS.get(model_name, model_name)}"
            )
            combo_fits.append(
                results[method][model_name]["iteration_counts"]["total_fits"]
            )
            combo_colors.append(method_colors[method])
    bars = ax.bar(range(len(combo_labels)), combo_fits, color=combo_colors)
    for bar, fits in zip(bars, combo_fits):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(fits),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(range(len(combo_labels)))
    ax.set_xticklabels(combo_labels, fontsize=7)
    ax.set_ylabel("Total Model Fits (candidates × CV folds)")
    ax.set_title("Number of Model Fits per (Method, Model)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[tune] saved comparison plot: {output_path}", flush=True)


def save_metrics(
    results: dict, metrics_path: Path, search_methods: list, model_names: list
):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    flat: dict = {}
    for method in search_methods:
        for model_name in model_names:
            r = results[method][model_name]
            prefix = f"{method}__{model_name}"
            flat[f"{prefix}__best_cv_f1_weighted"] = r["best_score"]
            flat[f"{prefix}__runtime_seconds"] = r["runtime_seconds"]
            flat[f"{prefix}__total_fits"] = r["iteration_counts"]["total_fits"]
            flat[f"{prefix}__n_candidates"] = r["iteration_counts"]["n_candidates"]
    with open(metrics_path, "w") as f:
        json.dump(flat, f, indent=2)
    print(f"[tune] saved metrics: {metrics_path}", flush=True)


@app.command()
def main(
    train_path: Path = typer.Option(...),
    test_path: Path = typer.Option(...),
    models_output_dir: Path = typer.Option(Path("data/models/tuned")),
    metrics_path: Path = typer.Option(Path("metrics/lab6_tuning.json")),
    plot_path: Path = typer.Option(Path("reports/lab6_tuning_comparison.png")),
    search_methods: str = typer.Option(
        "randomized_search,grid_search,halving_grid_search,halving_random_search"
    ),
    model_names: str = typer.Option("svm,random_forest,logistic_regression"),
    cv_folds: int = typer.Option(3),
    n_iter: int = typer.Option(20),
    sample_rows: int = typer.Option(30000),
    random_state: int = typer.Option(42),
    n_jobs: int = typer.Option(-1),
    max_features: int = typer.Option(20000),
    ngram_min: int = typer.Option(1),
    ngram_max: int = typer.Option(2),
    mlflow_enabled: bool = typer.Option(True),
    mlflow_tracking_uri: str = typer.Option("http://localhost:5001"),
    mlflow_experiment_name: str = typer.Option("lab6-hyperparameter-tuning"),
):
    methods_list = [m.strip() for m in search_methods.split(",") if m.strip()]
    models_list = [m.strip() for m in model_names.split(",") if m.strip()]

    print(f"[tune] loading train data from {train_path}", flush=True)
    df = _load_csv(train_path).dropna(subset=[TARGET_COLUMN]).copy()

    if sample_rows > 0 and sample_rows < len(df):
        df, _ = train_test_split(
            df,
            train_size=sample_rows,
            stratify=df[TARGET_COLUMN],
            random_state=random_state,
        )
        df = df.copy()

    text_column = _resolve_text_column(df)
    y_train = df[TARGET_COLUMN]
    X_train = df
    numeric_columns, categorical_columns = _split_columns(df, text_column)

    print(
        f"[tune] rows={len(df)} methods={methods_list} models={models_list}", flush=True
    )

    results = run_all_searches(
        X_train=X_train,
        y_train=y_train,
        text_column=text_column,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        search_methods=methods_list,
        model_names=models_list,
        cv=cv_folds,
        n_iter=n_iter,
        random_state=random_state,
        n_jobs=n_jobs,
        max_features=max_features,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        mlflow_enabled=mlflow_enabled,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
    )

    save_best_models(results, models_output_dir, models_list, methods_list)
    save_metrics(results, metrics_path, methods_list, models_list)
    plot_comparison(results, plot_path, methods_list, models_list)

    print("[tune] all done", flush=True)


if __name__ == "__main__":
    app()

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import joblib
import shap
import typer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))
from train_dummy_baseline import TARGET_COLUMN, _load_csv, _resolve_text_column

app = typer.Typer()


def load_pipeline_and_data(
    model_path: Path,
    train_path: Path,
    test_path: Path,
    sample_rows: int,
    random_state: int,
):
    pipeline = joblib.load(model_path)

    df_train = _load_csv(train_path).dropna(subset=[TARGET_COLUMN]).copy()
    if sample_rows > 0 and sample_rows < len(df_train):
        df_train, _ = train_test_split(
            df_train,
            train_size=sample_rows,
            stratify=df_train[TARGET_COLUMN],
            random_state=random_state,
        )
        df_train = df_train.copy()

    y_train = df_train[TARGET_COLUMN]
    X_train = df_train

    df_test = _load_csv(test_path).dropna(subset=[TARGET_COLUMN]).copy()
    y_test = df_test[TARGET_COLUMN]
    X_test = df_test

    return pipeline, X_train, y_train, X_test, y_test


def get_feature_names(preprocessor) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        names = []
        for name, transformer, _ in preprocessor.transformers_:
            try:
                names.extend(
                    [f"{name}__{f}" for f in transformer.get_feature_names_out()]
                )
            except Exception:
                pass
        return names


def get_explainer_and_transform(
    pipeline, X_background: pd.DataFrame, n_background: int
):
    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["model"]
    feature_names = get_feature_names(preprocessor)

    X_bg_transformed = preprocessor.transform(X_background.iloc[:n_background])

    if isinstance(classifier, (LinearSVC, LogisticRegression)):
        explainer = shap.LinearExplainer(
            classifier, X_bg_transformed, feature_perturbation="interventional"
        )
    elif isinstance(classifier, RandomForestClassifier):
        X_bg_dense = (
            X_bg_transformed.toarray()
            if hasattr(X_bg_transformed, "toarray")
            else X_bg_transformed
        )
        explainer = shap.TreeExplainer(
            classifier, X_bg_dense, feature_perturbation="tree_path_dependent"
        )
    else:
        explainer = shap.KernelExplainer(
            classifier.predict_proba
            if hasattr(classifier, "predict_proba")
            else classifier.decision_function,
            shap.sample(X_bg_transformed, n_background),
        )

    return explainer, preprocessor, feature_names


def compute_shap_values(
    explainer,
    preprocessor,
    X_explain: pd.DataFrame,
    classifier,
    n_explain: int,
) -> tuple[np.ndarray, np.ndarray]:
    X_sub = X_explain.iloc[:n_explain]
    X_transformed = preprocessor.transform(X_sub)

    if isinstance(classifier, RandomForestClassifier):
        X_dense = (
            X_transformed.toarray()
            if hasattr(X_transformed, "toarray")
            else X_transformed
        )
        raw = explainer.shap_values(X_dense, check_additivity=False)
    else:
        raw = explainer.shap_values(X_transformed)

    if isinstance(raw, list):
        shap_array = np.stack(raw, axis=-1)
    elif isinstance(raw, np.ndarray) and raw.ndim == 3:
        shap_array = raw
    else:
        shap_array = raw[:, :, np.newaxis]

    X_dense_out = (
        X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed
    )
    return shap_array, X_dense_out


def get_top_feature_idx(shap_array: np.ndarray, top_n: int) -> np.ndarray:
    mean_abs = np.abs(shap_array).mean(axis=(0, 2))
    return np.argsort(mean_abs)[-top_n:][::-1]


def plot_summary(
    shap_array: np.ndarray,
    X_dense: np.ndarray,
    feature_names: list,
    class_labels: list,
    output_dir: Path,
    search_method: str,
    top_n: int = 30,
):
    n_classes = shap_array.shape[2]
    for class_idx in tqdm(range(n_classes), desc="SHAP summary plots", unit="class"):
        label = class_labels[class_idx] if class_idx < len(class_labels) else class_idx
        shap_vals = shap_array[:, :, class_idx]

        top_idx = np.argsort(np.abs(shap_vals).mean(axis=0))[-top_n:][::-1]
        shap_top = shap_vals[:, top_idx]
        X_top = X_dense[:, top_idx]
        feat_top = [feature_names[i] for i in top_idx]

        plt.figure()
        shap.summary_plot(
            shap_top, X_top, feature_names=feat_top, show=False, max_display=top_n
        )
        plt.title(f"SHAP Summary — class {label} | {search_method}")
        plt.tight_layout()
        out_path = output_dir / f"shap_summary_class{label}.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[shap] saved {out_path}", flush=True)


def plot_feature_importance_bar(
    shap_array: np.ndarray,
    feature_names: list,
    output_dir: Path,
    search_method: str,
    top_n: int = 20,
):
    mean_abs = np.abs(shap_array).mean(axis=(0, 2))
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_vals = mean_abs[top_idx]

    colors = []
    for name in top_names:
        if name.startswith("text__"):
            colors.append("#4e79a7")
        elif name.startswith("num__"):
            colors.append("#f28e2b")
        else:
            colors.append("#76b7b2")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(top_names)), top_vals[::-1], color=colors[::-1])
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance (SHAP) | {search_method}")

    from matplotlib.patches import Patch

    legend_elems = [
        Patch(color="#4e79a7", label="text"),
        Patch(color="#f28e2b", label="numeric"),
        Patch(color="#76b7b2", label="categorical"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
    fig.tight_layout()
    out_path = output_dir / f"shap_feature_importance.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[shap] saved {out_path}", flush=True)


def find_misclassified(
    pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_examples: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    y_pred = pd.Series(pipeline.predict(X_test), index=y_test.index)
    wrong_mask = y_pred != y_test
    X_wrong = X_test[wrong_mask]
    y_true_wrong = y_test[wrong_mask]
    y_pred_wrong = y_pred[wrong_mask]

    rng = np.random.RandomState(random_state)
    n = min(n_examples, len(X_wrong))
    idx = rng.choice(len(X_wrong), size=n, replace=False)
    return X_wrong.iloc[idx], y_true_wrong.iloc[idx], y_pred_wrong.iloc[idx]


def plot_misclassified_waterfall(
    explainer,
    preprocessor,
    classifier,
    X_wrong: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    feature_names: list,
    class_labels: list,
    output_dir: Path,
    search_method: str,
    n_plots: int = 5,
):
    n = min(n_plots, len(X_wrong))
    for i in tqdm(range(n), desc="Misclassified waterfall plots", unit="example"):
        row = X_wrong.iloc[[i]]
        true_label = y_true.iloc[i]
        pred_label = y_pred.iloc[i]

        X_transformed = preprocessor.transform(row)
        if isinstance(classifier, RandomForestClassifier):
            X_dense = (
                X_transformed.toarray()
                if hasattr(X_transformed, "toarray")
                else X_transformed
            )
            raw = explainer.shap_values(X_dense, check_additivity=False)
        else:
            raw = explainer.shap_values(X_transformed)

        if isinstance(raw, list):
            shap_vals_row = raw
        elif isinstance(raw, np.ndarray) and raw.ndim == 3:
            shap_vals_row = [raw[:, :, c] for c in range(raw.shape[2])]
        else:
            shap_vals_row = [raw]

        pred_class_idx = (
            class_labels.index(int(pred_label))
            if int(pred_label) in class_labels
            else 0
        )
        sv = shap_vals_row[pred_class_idx][0]

        top_idx = np.argsort(np.abs(sv))[-15:][::-1]
        top_sv = sv[top_idx]
        top_names = [feature_names[j] for j in top_idx]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#e15759" if v < 0 else "#4e79a7" for v in top_sv]
        ax.barh(range(len(top_names)), top_sv[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=8)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value")

        text_col = (
            "review_text_clean"
            if "review_text_clean" in X_wrong.columns
            else "review_text"
        )
        snippet = (
            str(X_wrong.iloc[i].get(text_col, ""))[:120]
            if text_col in X_wrong.columns
            else ""
        )
        ax.set_title(
            f'Misclassified #{i + 1} | True: {true_label} → Pred: {pred_label} | {search_method}\n"{snippet}..."',
            fontsize=9,
        )
        fig.tight_layout()
        out_path = output_dir / f"shap_misclassified_{i}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[shap] saved {out_path}", flush=True)


def save_analysis_json(
    shap_array: np.ndarray,
    feature_names: list,
    class_labels: list,
    X_wrong: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    output_dir: Path,
    search_method: str,
):
    analysis = {"search_method": search_method, "classes": {}}

    for class_idx, label in enumerate(class_labels):
        sv = shap_array[:, :, class_idx]
        mean_sv = sv.mean(axis=0)

        pos_idx = np.argsort(mean_sv)[-10:][::-1]
        neg_idx = np.argsort(mean_sv)[:10]

        analysis["classes"][str(label)] = {
            "top_positive_features": [
                {"feature": feature_names[j], "mean_shap": float(mean_sv[j])}
                for j in pos_idx
            ],
            "top_negative_features": [
                {"feature": feature_names[j], "mean_shap": float(mean_sv[j])}
                for j in neg_idx
            ],
        }

    text_col = (
        "review_text_clean" if "review_text_clean" in X_wrong.columns else "review_text"
    )
    misclassified_list = []
    for i in range(len(X_wrong)):
        snippet = (
            str(X_wrong.iloc[i].get(text_col, ""))[:200]
            if text_col in X_wrong.columns
            else ""
        )
        misclassified_list.append(
            {
                "true_label": int(y_true.iloc[i]),
                "predicted_label": int(y_pred.iloc[i]),
                "text_snippet": snippet,
            }
        )
    analysis["misclassified_examples"] = misclassified_list

    out_path = output_dir / "shap_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"[shap] saved {out_path}", flush=True)


@app.command()
def main(
    model_path: Path = typer.Option(..., "--model-path"),
    search_method: str = typer.Option(..., "--search-method"),
    train_path: Path = typer.Option(Path("data/split/train_reviews.csv")),
    test_path: Path = typer.Option(Path("data/split/test_reviews.csv")),
    output_dir: Path = typer.Option(Path("reports/shap")),
    metrics_path: Path = typer.Option(..., "--metrics-path"),
    sample_rows_background: int = typer.Option(200),
    sample_rows_explain: int = typer.Option(500),
    n_misclassified: int = typer.Option(10),
    top_n_features: int = typer.Option(30),
    random_state: int = typer.Option(42),
    mlflow_enabled: bool = typer.Option(True),
    mlflow_tracking_uri: str = typer.Option("http://localhost:5001"),
    mlflow_experiment_name: str = typer.Option("lab6-shap-explanations"),
):
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    steps = [
        "load data & pipeline",
        "build explainer",
        "compute SHAP values",
        "summary plots",
        "feature importance plot",
        "find misclassified",
        "waterfall plots",
        "save JSON",
    ]
    pbar = tqdm(steps, desc=f"SHAP [{search_method}]", unit="step")

    pbar.set_description(f"[1/8] loading pipeline & data")
    pipeline, X_train, y_train, X_test, y_test = load_pipeline_and_data(
        model_path,
        train_path,
        test_path,
        sample_rows=max(sample_rows_background, sample_rows_explain) + 100,
        random_state=random_state,
    )
    pbar.update(1)

    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["model"]

    pbar.set_description(
        f"[2/8] building explainer (background={sample_rows_background})"
    )
    explainer, preprocessor, feature_names = get_explainer_and_transform(
        pipeline, X_train, n_background=sample_rows_background
    )
    pbar.update(1)

    pbar.set_description(f"[3/8] computing SHAP values (n={sample_rows_explain})")
    shap_array, X_dense = compute_shap_values(
        explainer, preprocessor, X_train, classifier, n_explain=sample_rows_explain
    )
    pbar.update(1)

    class_labels = sorted(int(x) for x in y_train.unique())
    tqdm.write(
        f"[shap] classes={class_labels}  shape={shap_array.shape}  features={len(feature_names)}"
    )

    pbar.set_description("[4/8] summary plots (per class)")
    plot_summary(
        shap_array,
        X_dense,
        feature_names,
        class_labels,
        output_dir,
        search_method,
        top_n=top_n_features,
    )
    pbar.update(1)

    pbar.set_description("[5/8] feature importance bar plot")
    plot_feature_importance_bar(
        shap_array,
        feature_names,
        output_dir,
        search_method,
        top_n=min(20, top_n_features),
    )
    pbar.update(1)

    pbar.set_description("[6/8] finding misclassified examples")
    X_wrong, y_true_wrong, y_pred_wrong = find_misclassified(
        pipeline, X_test, y_test, n_examples=n_misclassified, random_state=random_state
    )
    tqdm.write(f"[shap] found {len(X_wrong)} misclassified examples")
    pbar.update(1)

    pbar.set_description("[7/8] waterfall plots for misclassified")
    plot_misclassified_waterfall(
        explainer,
        preprocessor,
        classifier,
        X_wrong,
        y_true_wrong,
        y_pred_wrong,
        feature_names,
        class_labels,
        output_dir,
        search_method,
        n_plots=5,
    )
    pbar.update(1)

    pbar.set_description("[8/8] saving analysis JSON")
    save_analysis_json(
        shap_array,
        feature_names,
        class_labels,
        X_wrong,
        y_true_wrong,
        y_pred_wrong,
        output_dir,
        search_method,
    )
    pbar.update(1)
    pbar.close()

    metrics = {
        "search_method": search_method,
        "n_features": int(shap_array.shape[1]),
        "n_explained_samples": int(shap_array.shape[0]),
        "n_misclassified_analyzed": int(len(X_wrong)),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[shap] saved metrics: {metrics_path}", flush=True)

    if mlflow_enabled:
        try:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(mlflow_experiment_name)
            with mlflow.start_run(run_name=f"shap_{search_method}"):
                mlflow.set_tag("search_method", search_method)
                mlflow.log_metric("n_features", metrics["n_features"])
                mlflow.log_metric("n_explained_samples", metrics["n_explained_samples"])
                mlflow.log_artifacts(str(output_dir), artifact_path="shap_plots")
        except Exception as e:
            print(f"[shap] mlflow logging failed: {e}", flush=True)

    print(f"[shap] done for {search_method}", flush=True)


if __name__ == "__main__":
    app()

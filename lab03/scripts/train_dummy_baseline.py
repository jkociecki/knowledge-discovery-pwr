import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
import typer
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

TARGET_COLUMN = "LABEL-simple_rating"
TEXT_COLUMN = "review_text"
TEXT_CLEAN_COLUMN = "review_text_clean"
SCORING = {
    "accuracy": "accuracy",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
}

TEXT_VECTORIZERS = {"bow", "tfidf", "word2vec"}
FEATURE_SELECTIONS = {"none", "select_k_best"}
DIMENSION_REDUCTIONS = {"none", "svd"}
WORD2VEC_CACHE_DIR = Path("data/models/word2vec")
WORD2VEC_MODEL_NAME = "google-news-300"


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_")


def _load_word2vec_model(cache_dir: Path = WORD2VEC_CACHE_DIR):
    import gensim.downloader as api

    cache_dir.mkdir(parents=True, exist_ok=True)
    model_name = "word2vec-google-news-300"
    return api.load(model_name)


def _mean_word2vec_vector(text: str, model, vector_size: int) -> np.ndarray:
    tokens = [token.lower() for token in re.findall(r"[a-zA-Z]+", text or "")]
    vectors = []
    for token in tokens:
        if token in model:
            vectors.append(model[token])

    if vectors:
        return np.mean(np.asarray(vectors), axis=0)
    return np.zeros(vector_size, dtype=np.float32)


class Word2VecVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, cache_dir: Path = WORD2VEC_CACHE_DIR):
        self.cache_dir = cache_dir
        self.model = None
        self.vector_size = 300

    def fit(self, X, y=None):
        if not hasattr(self, "model_"):
            self.model_ = _load_word2vec_model(self.cache_dir)
            self.vector_size_ = int(self.model_.vector_size)
            self.model = self.model_
            self.vector_size = self.vector_size_
        return self

    def transform(self, X):
        if not hasattr(self, "model_"):
            self.fit(X)

        texts = pd.Series(np.asarray(X).ravel()).fillna("").astype(str).tolist()
        vectors = [
            _mean_word2vec_vector(text, self.model_, self.vector_size_)
            for text in texts
        ]
        return np.asarray(vectors, dtype=np.float32)


def _compute_scores(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _summarize_scores(score_values: dict[str, list[float]]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for metric_name, values in score_values.items():
        series = np.array(values, dtype=float)
        summary[metric_name] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "fold_scores": [float(item) for item in series.tolist()],
        }
    return summary


def _build_confusion_matrix_figure(
    cm: np.ndarray,
    labels: list[int],
    title: str,
):
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for row_idx in range(cm.shape[0]):
        for col_idx in range(cm.shape[1]):
            color = "white" if cm[row_idx, col_idx] > cm.max() * 0.5 else "black"
            ax.text(
                col_idx,
                row_idx,
                cm[row_idx, col_idx],
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    return fig


def _build_text_vectorizer(
    vectorizer_name: str,
    ngram_min: int,
    ngram_max: int,
    max_features: int,
):
    if vectorizer_name == "bow":
        return CountVectorizer(
            ngram_range=(ngram_min, ngram_max),
            max_features=max_features,
        )
    if vectorizer_name == "tfidf":
        return TfidfVectorizer(
            ngram_range=(ngram_min, ngram_max),
            max_features=max_features,
        )
    if vectorizer_name == "word2vec":
        return Word2VecVectorizer()
    raise ValueError(f"Unsupported vectorizer_name: {vectorizer_name}")


def _build_text_postprocessor(
    vectorizer_name: str,
    feature_selection: str,
    dimension_reduction: str,
    select_k: int,
    svd_components: int,
    random_state: int,
):
    steps = []

    if feature_selection == "select_k_best":
        score_func = chi2 if vectorizer_name in {"bow", "tfidf"} else f_classif
        steps.append(("select_k_best", SelectKBest(score_func=score_func, k=select_k)))

    if dimension_reduction == "svd":
        steps.append(
            (
                "svd",
                TruncatedSVD(n_components=svd_components, random_state=random_state),
            )
        )

    return Pipeline(steps=steps) if steps else None


def _load_csv(path: Path) -> pd.DataFrame:
    frame = pl.read_csv(
        path,
        schema_overrides={
            TARGET_COLUMN: pl.Int32,
        },
        encoding="utf8-lossy",
        truncate_ragged_lines=True,
        ignore_errors=True,
    )
    return frame.to_pandas()


def _resolve_feature_sets(feature_sets: str) -> list[str]:
    supported = {"text_only", "non_text", "all"}
    parsed = [item.strip() for item in feature_sets.split(",") if item.strip()]
    unique = list(dict.fromkeys(parsed))

    invalid = [item for item in unique if item not in supported]
    if invalid:
        raise ValueError(f"Unsupported feature sets: {invalid}")

    return unique or ["all"]


def _resolve_models(model_name: str) -> list[str]:
    if model_name == "all":
        return ["dummy", "svm", "random_forest"]

    aliases = {
        "dummy": "dummy",
        "svm": "svm",
        "random_forest": "random_forest",
        "rf": "random_forest",
    }
    if model_name not in aliases:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return [aliases[model_name]]


def _resolve_text_column(df: pd.DataFrame) -> str:
    if TEXT_CLEAN_COLUMN in df.columns:
        return TEXT_CLEAN_COLUMN
    if TEXT_COLUMN in df.columns:
        return TEXT_COLUMN
    raise ValueError(
        f"Expected one of text columns: {TEXT_CLEAN_COLUMN!r} or {TEXT_COLUMN!r}"
    )


def _split_columns(df: pd.DataFrame, text_column: str) -> tuple[list[str], list[str]]:
    all_feature_columns = [
        column
        for column in df.columns
        if column not in {TARGET_COLUMN, TEXT_COLUMN, TEXT_CLEAN_COLUMN, text_column}
    ]

    numeric_columns = [
        column
        for column in all_feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
        or pd.api.types.is_bool_dtype(df[column])
    ]
    categorical_columns = [
        column for column in all_feature_columns if column not in numeric_columns
    ]
    return numeric_columns, categorical_columns


def _build_preprocessor(
    feature_set: str,
    text_column: str,
    numeric_columns: list[str],
    categorical_columns: list[str],
    vectorizer_name: str,
    ngram_min: int,
    ngram_max: int,
    max_features: int,
    feature_selection: str,
    dimension_reduction: str,
    select_k: int,
    svd_components: int,
    random_state: int,
):
    text_steps = [
        (
            "vectorizer",
            _build_text_vectorizer(vectorizer_name, ngram_min, ngram_max, max_features),
        )
    ]
    text_postprocessor = _build_text_postprocessor(
        vectorizer_name=vectorizer_name,
        feature_selection=feature_selection,
        dimension_reduction=dimension_reduction,
        select_k=select_k,
        svd_components=svd_components,
        random_state=random_state,
    )
    if text_postprocessor is not None:
        text_steps.append(("postprocess", text_postprocessor))

    text_pipeline = Pipeline(steps=text_steps)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    transformers = []
    if feature_set in {"text_only", "all"}:
        transformers.append(("text", text_pipeline, text_column))
    if feature_set in {"non_text", "all"}:
        if numeric_columns:
            transformers.append(("num", numeric_pipeline, numeric_columns))
        if categorical_columns:
            transformers.append(("cat", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError(f"No transformers configured for feature_set={feature_set}")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_estimator(
    model: str,
    model_strategy: str,
    random_state: int,
    n_estimators: int,
):
    if model == "dummy":
        return DummyClassifier(strategy=model_strategy, random_state=random_state)
    if model == "svm":
        return LinearSVC(random_state=random_state)
    if model == "random_forest":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model}")


def train_and_evaluate(
    train_path: Path,
    test_path: Path,
    metrics_path: Path,
    model_name: str = "all",
    feature_sets: str = "text_only,non_text,all",
    vectorizer_name: str = "bow",
    model_strategy: str = "most_frequent",
    random_state: int = 42,
    n_estimators: int = 200,
    max_features: int = 20000,
    ngram_min: int = 1,
    ngram_max: int = 2,
    feature_selection: str = "none",
    dimension_reduction: str = "none",
    select_k: int = 1000,
    svd_components: int = 300,
    cv_folds: int = 5,
    language_model: str = "en_core_web_sm",
    sample_rows: int = 0,
    mlflow_enabled: bool = True,
    mlflow_log_confusion_matrices: bool = True,
    mlflow_log_model: bool = True,
    mlflow_tracking_uri: str = "http://host.docker.internal:5001",
    mlflow_experiment_name: str = "lab5-sentiment",
    mlflow_run_prefix: str = "baseline",
) -> dict:
    df = _load_csv(train_path)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()

    if sample_rows > 0 and sample_rows < len(df):
        n_classes = df[TARGET_COLUMN].nunique(dropna=True)
        if sample_rows < n_classes:
            raise ValueError(
                "sample_rows must be at least the number of target classes "
                f"({n_classes}), got {sample_rows}"
            )

        sampled_df, _ = train_test_split(
            df,
            train_size=sample_rows,
            stratify=df[TARGET_COLUMN],
            random_state=random_state,
        )
        df = sampled_df.copy()

    text_column = _resolve_text_column(df)
    y = df[TARGET_COLUMN]
    X = df

    if vectorizer_name not in TEXT_VECTORIZERS:
        raise ValueError(f"Unsupported vectorizer_name: {vectorizer_name}")
    if feature_selection not in FEATURE_SELECTIONS:
        raise ValueError(f"Unsupported feature_selection: {feature_selection}")
    if dimension_reduction not in DIMENSION_REDUCTIONS:
        raise ValueError(f"Unsupported dimension_reduction: {dimension_reduction}")
    if feature_selection == "select_k_best" and select_k <= 0:
        raise ValueError(
            "select_k must be positive when feature_selection=select_k_best"
        )
    if dimension_reduction == "svd" and svd_components <= 0:
        raise ValueError("svd_components must be positive when dimension_reduction=svd")

    test_df: pd.DataFrame | None = None
    if test_path.exists():
        loaded_test = _load_csv(test_path).dropna(subset=[TARGET_COLUMN]).copy()
        if len(loaded_test) > 0:
            test_df = loaded_test

    numeric_columns, categorical_columns = _split_columns(df, text_column=text_column)
    selected_feature_sets = _resolve_feature_sets(feature_sets)
    selected_models = _resolve_models(model_name)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    class_labels = sorted(int(label) for label in y.unique())

    mlflow_module = None
    if mlflow_enabled:
        mlflow_module = mlflow
        mlflow_module.set_tracking_uri(mlflow_tracking_uri)
        mlflow_module.set_experiment(mlflow_experiment_name)

    total_runs = len(selected_feature_sets) * len(selected_models)
    run_idx = 0
    print(
        f"[train] rows={len(df)} cv_folds={cv_folds} runs={total_runs} sample_rows={sample_rows}",
        flush=True,
    )

    results: dict[str, dict[str, dict]] = {}
    for feature_set in selected_feature_sets:
        preprocessor = _build_preprocessor(
            feature_set=feature_set,
            text_column=text_column,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            vectorizer_name=vectorizer_name,
            ngram_min=ngram_min,
            ngram_max=ngram_max,
            max_features=max_features,
            feature_selection=feature_selection,
            dimension_reduction=dimension_reduction,
            select_k=select_k,
            svd_components=svd_components,
            random_state=random_state,
        )

        results[feature_set] = {}
        for model in selected_models:
            run_idx += 1
            run_started = time.perf_counter()
            print(
                f"[train] ({run_idx}/{total_runs}) start feature_set={feature_set} model={model}",
                flush=True,
            )

            estimator = _build_estimator(
                model=model,
                model_strategy=model_strategy,
                random_state=random_state,
                n_estimators=n_estimators,
            )

            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", estimator),
                ]
            )

            cv_train_scores: dict[str, list[float]] = {metric: [] for metric in SCORING}
            cv_validation_scores: dict[str, list[float]] = {
                metric: [] for metric in SCORING
            }
            cm_train = np.zeros((len(class_labels), len(class_labels)), dtype=int)
            cm_validation = np.zeros((len(class_labels), len(class_labels)), dtype=int)

            for train_idx, validation_idx in cv.split(X, y):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_validation_fold = X.iloc[validation_idx]
                y_validation_fold = y.iloc[validation_idx]

                fold_pipeline = clone(pipeline)
                fold_pipeline.fit(X_train_fold, y_train_fold)

                y_train_pred = fold_pipeline.predict(X_train_fold)
                y_validation_pred = fold_pipeline.predict(X_validation_fold)

                train_scores = _compute_scores(y_train_fold, y_train_pred)
                validation_scores = _compute_scores(
                    y_validation_fold, y_validation_pred
                )
                for metric_name in SCORING:
                    cv_train_scores[metric_name].append(train_scores[metric_name])
                    cv_validation_scores[metric_name].append(
                        validation_scores[metric_name]
                    )

                cm_train += confusion_matrix(
                    y_train_fold,
                    y_train_pred,
                    labels=class_labels,
                )
                cm_validation += confusion_matrix(
                    y_validation_fold,
                    y_validation_pred,
                    labels=class_labels,
                )

            full_pipeline = clone(pipeline)
            full_pipeline.fit(X, y)
            y_train_full_pred = full_pipeline.predict(X)
            cm_train_full = confusion_matrix(y, y_train_full_pred, labels=class_labels)

            test_scores: dict[str, float] | None = None
            test_cm_path: Path | None = None
            if test_df is not None:
                test_text_column = _resolve_text_column(test_df)
                if test_text_column != text_column and text_column in test_df.columns:
                    test_df[text_column] = test_df[test_text_column]
                y_test = test_df[TARGET_COLUMN]
                X_test = test_df
                y_test_pred = full_pipeline.predict(X_test)
                test_scores = _compute_scores(y_test, y_test_pred)
                cm_test = confusion_matrix(y_test, y_test_pred, labels=class_labels)
            else:
                cm_test = None

            elapsed = time.perf_counter() - run_started
            print(
                f"[train] ({run_idx}/{total_runs}) done feature_set={feature_set} model={model} "
                f"in {elapsed:.1f}s",
                flush=True,
            )

            run_summary = {
                "train_cv": _summarize_scores(cv_train_scores),
                "validation_cv": _summarize_scores(cv_validation_scores),
                "test": test_scores,
                "mlflow": {
                    "model_logged": False,
                    "confusion_matrices_logged": False,
                },
            }

            if mlflow_module is not None:
                run_name = f"{mlflow_run_prefix}-feature_{_safe_name(feature_set)}-model_{_safe_name(model)}"
                try:
                    with mlflow_module.start_run(run_name=run_name):
                        mlflow_module.log_params(
                            {
                                "feature_set": feature_set,
                                "vectorizer_name": vectorizer_name,
                                "feature_selection": feature_selection,
                                "dimension_reduction": dimension_reduction,
                                "model": model,
                                "model_strategy": model_strategy,
                                "random_state": random_state,
                                "n_estimators": n_estimators,
                                "max_features": max_features,
                                "ngram_min": ngram_min,
                                "ngram_max": ngram_max,
                                "cv_folds": cv_folds,
                                "language_model": language_model,
                                "sample_rows": sample_rows,
                                "mlflow_log_confusion_matrices": mlflow_log_confusion_matrices,
                                "mlflow_log_model": mlflow_log_model,
                            }
                        )

                        for metric_name in SCORING:
                            mlflow_module.log_metric(
                                f"cv_train_{metric_name}_mean",
                                run_summary["train_cv"][metric_name]["mean"],
                            )
                            mlflow_module.log_metric(
                                f"cv_validation_{metric_name}_mean",
                                run_summary["validation_cv"][metric_name]["mean"],
                            )
                            mlflow_module.log_metric(
                                f"cv_validation_{metric_name}_std",
                                run_summary["validation_cv"][metric_name]["std"],
                            )

                        if run_summary["test"] is not None:
                            for metric_name, metric_value in run_summary[
                                "test"
                            ].items():
                                mlflow_module.log_metric(
                                    f"test_{metric_name}", metric_value
                                )

                        if mlflow_log_confusion_matrices:
                            cm_figures = [
                                (
                                    cm_train,
                                    f"Train CM (CV folds aggregated) - {feature_set}/{model}",
                                    "confusion_matrices/cm_train_cv.png",
                                ),
                                (
                                    cm_validation,
                                    f"Validation CM (CV folds aggregated) - {feature_set}/{model}",
                                    "confusion_matrices/cm_validation_cv.png",
                                ),
                                (
                                    cm_train_full,
                                    f"Train CM (full train fit) - {feature_set}/{model}",
                                    "confusion_matrices/cm_train_full.png",
                                ),
                            ]
                            if cm_test is not None:
                                cm_figures.append(
                                    (
                                        cm_test,
                                        f"Test CM - {feature_set}/{model}",
                                        "confusion_matrices/cm_test.png",
                                    )
                                )

                            for cm_matrix, cm_title, cm_artifact_path in cm_figures:
                                fig = _build_confusion_matrix_figure(
                                    cm=cm_matrix,
                                    labels=class_labels,
                                    title=cm_title,
                                )
                                mlflow_module.log_figure(fig, cm_artifact_path)
                                plt.close(fig)
                            run_summary["mlflow"]["confusion_matrices_logged"] = True

                        if mlflow_log_model:
                            try:
                                mlflow_module.sklearn.log_model(
                                    sk_model=full_pipeline,
                                    artifact_path="model",
                                )
                                run_summary["mlflow"]["model_logged"] = True
                            except Exception as exc:  # noqa: BLE001
                                print(
                                    f"[warn] failed to log model to MLflow: {exc}",
                                    flush=True,
                                )

                        mlflow_module.log_dict(run_summary, "run_summary.json")
                except Exception as exc:  # noqa: BLE001
                    # Keep training reproducible even when MLflow artifact store is unavailable.
                    print(
                        f"[warn] MLflow run logging failed for {feature_set}/{model}: {exc}",
                        flush=True,
                    )

            results[feature_set][model] = run_summary

    metrics = {
        "target_column": TARGET_COLUMN,
        "sample_rows": int(sample_rows),
        "rows_used": int(len(df)),
        "test_rows": int(len(test_df)) if test_df is not None else 0,
        "cv_folds": cv_folds,
        "mlflow": {
            "enabled": bool(mlflow_module is not None),
            "tracking_uri": mlflow_tracking_uri,
            "experiment_name": mlflow_experiment_name,
        },
        "feature_columns": {
            "text": [text_column],
            "numeric": numeric_columns,
            "categorical": categorical_columns,
        },
        "text_pipeline": {
            "vectorizer_name": vectorizer_name,
            "feature_selection": feature_selection,
            "dimension_reduction": dimension_reduction,
            "select_k": select_k,
            "svd_components": svd_components,
        },
        "experiments": results,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(
    train_path: Path = typer.Option(
        Path("data/preprocessed/preprocessed_reviews.csv"), "--train-path"
    ),
    test_path: Path = typer.Option(
        Path("data/split/test_reviews.csv"),
        "--test-path",
        help="Path to holdout test split used for final evaluation and test confusion matrix.",
    ),
    metrics_path: Path = typer.Option(
        Path("metrics/model_cv_metrics.json"), "--metrics-path"
    ),
    model_name: str = typer.Option("all", "--model-name"),
    model_strategy: str = typer.Option("most_frequent", "--model-strategy"),
    feature_set: str = typer.Option(
        "text_only,non_text,all",
        "--feature-set",
        help="Comma-separated values: text_only, non_text, all",
    ),
    vectorizer_name: str = typer.Option("bow", "--vectorizer-name"),
    random_state: int = typer.Option(42, "--random-state"),
    n_estimators: int = typer.Option(200, "--n-estimators"),
    max_features: int = typer.Option(20000, "--max-features"),
    ngram_min: int = typer.Option(1, "--ngram-min"),
    ngram_max: int = typer.Option(2, "--ngram-max"),
    feature_selection: str = typer.Option("none", "--feature-selection"),
    dimension_reduction: str = typer.Option("none", "--dimension-reduction"),
    select_k: int = typer.Option(1000, "--select-k"),
    svd_components: int = typer.Option(300, "--svd-components"),
    cv_folds: int = typer.Option(5, "--cv-folds"),
    language_model: str = typer.Option("en_core_web_sm", "--language-model"),
    sample_rows: int = typer.Option(
        0,
        "--sample-rows",
        help="Limit number of rows used in CV for faster experiments. 0 means all rows.",
    ),
    mlflow_enabled: bool = typer.Option(True, "--mlflow-enabled/--no-mlflow-enabled"),
    mlflow_log_confusion_matrices: bool = typer.Option(
        True,
        "--mlflow-log-confusion-matrices/--no-mlflow-log-confusion-matrices",
    ),
    mlflow_log_model: bool = typer.Option(
        True, "--mlflow-log-model/--no-mlflow-log-model"
    ),
    mlflow_tracking_uri: str = typer.Option(
        "http://host.docker.internal:5001",
        "--mlflow-tracking-uri",
    ),
    mlflow_experiment_name: str = typer.Option(
        "lab5-sentiment",
        "--mlflow-experiment-name",
    ),
    mlflow_run_prefix: str = typer.Option("baseline", "--mlflow-run-prefix"),
) -> None:
    metrics = train_and_evaluate(
        train_path=train_path,
        test_path=test_path,
        metrics_path=metrics_path,
        model_name=model_name,
        feature_sets=feature_set,
        vectorizer_name=vectorizer_name,
        model_strategy=model_strategy,
        random_state=random_state,
        n_estimators=n_estimators,
        max_features=max_features,
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        feature_selection=feature_selection,
        dimension_reduction=dimension_reduction,
        select_k=select_k,
        svd_components=svd_components,
        cv_folds=cv_folds,
        language_model=language_model,
        sample_rows=sample_rows,
        mlflow_enabled=mlflow_enabled,
        mlflow_log_confusion_matrices=mlflow_log_confusion_matrices,
        mlflow_log_model=mlflow_log_model,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_prefix=mlflow_run_prefix,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    typer.run(main)

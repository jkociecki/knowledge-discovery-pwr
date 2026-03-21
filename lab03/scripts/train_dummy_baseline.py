import json
from pathlib import Path

import numpy as np
import polars as pl
import typer
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score


def _load_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        schema_overrides={
            "LABEL-simple_rating": pl.Int32,
            "review_text": pl.Utf8,
            "review_title": pl.Utf8,
        },
        encoding="utf8-lossy",
        truncate_ragged_lines=True,
        ignore_errors=True,
    )


def _build_text_feature(df: pl.DataFrame) -> list[str]:
    return (
        df.with_columns(
            pl.concat_str(
                [
                    pl.col("review_title").fill_null(""),
                    pl.col("review_text").fill_null(""),
                ],
                separator=" ",
            ).alias("text_feature")
        )
        .get_column("text_feature")
        .to_list()
    )


def train_and_evaluate(
    train_path: Path,
    test_path: Path,
    metrics_path: Path,
    model_type: str = "dummy",
    model_strategy: str = "most_frequent",
    max_features: int = 20000,
    ngram_max: int = 2,
) -> dict:
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Missing split files. Run scripts/train_test_split.py first to create data/split/train_reviews.csv and data/split/test_reviews.csv."
        )

    train_df = _load_csv(train_path).drop_nulls(["LABEL-simple_rating"])
    test_df = _load_csv(test_path).drop_nulls(["LABEL-simple_rating"])

    x_train = _build_text_feature(train_df)
    x_test = _build_text_feature(test_df)

    y_train = train_df.get_column("LABEL-simple_rating").to_list()
    y_test = test_df.get_column("LABEL-simple_rating").to_list()

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, ngram_max))
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    if model_type == "dummy":
        model = DummyClassifier(strategy=model_strategy, random_state=42)
        model.fit(x_train_vec, y_train)
        y_pred = model.predict(x_test_vec)
        model_name = f"DummyClassifier(strategy={model_strategy})"

    elif model_type == "uniform_continuous":
        classes = sorted(set(y_train))
        if not classes:
            raise ValueError("No target classes found in train set.")

        rng = np.random.default_rng(42)
        random_values = rng.uniform(0.0, 1.0, size=len(y_test))
        bin_edges = np.linspace(0.0, 1.0, num=len(classes) + 1)
        class_indices = np.digitize(random_values, bin_edges[1:-1], right=False)
        y_pred = np.array([classes[idx] for idx in class_indices])
        model_name = "UniformContinuousSampler"

    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    metrics = {
        "model": model_name,
        "model_type": model_type,
        "model_strategy": model_strategy,
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "target_column": "LABEL-simple_rating",
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main(
    train_path: Path = typer.Option(
        Path("data/split/train_reviews.csv"), "--train-path"
    ),
    test_path: Path = typer.Option(Path("data/split/test_reviews.csv"), "--test-path"),
    metrics_path: Path = typer.Option(
        Path("metrics/dummy_baseline_metrics.json"), "--metrics-path"
    ),
    model_type: str = typer.Option("dummy", "--model-type"),
    model_strategy: str = typer.Option("most_frequent", "--model-strategy"),
    max_features: int = typer.Option(20000, "--max-features"),
    ngram_max: int = typer.Option(2, "--ngram-max"),
) -> None:
    metrics = train_and_evaluate(
        train_path=train_path,
        test_path=test_path,
        metrics_path=metrics_path,
        model_type=model_type,
        model_strategy=model_strategy,
        max_features=max_features,
        ngram_max=ngram_max,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    typer.run(main)

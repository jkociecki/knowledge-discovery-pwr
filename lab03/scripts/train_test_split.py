from pathlib import Path

import polars as pl
from sklearn.model_selection import train_test_split
import typer


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def split_data(
    reviews_path: str,
    train_path: str,
    test_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = False,
) -> None:
    reviews_df = pl.read_csv(
        reviews_path,
        schema_overrides={
            "LABEL-simple_rating": pl.Int32,
        },
        truncate_ragged_lines=True,
        ignore_errors=True,
    )

    reviews_df = reviews_df.with_row_index("row_id")

    stratify_labels = None
    if stratify:
        stratify_labels = reviews_df["LABEL-simple_rating"].to_list()

    train_ids, test_ids = train_test_split(
        reviews_df["row_id"].to_list(),
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    train_df = reviews_df.filter(pl.col("row_id").is_in(train_ids)).drop("row_id")
    test_df = reviews_df.filter(pl.col("row_id").is_in(test_ids)).drop("row_id")

    train_output = Path(train_path)
    test_output = Path(test_path)
    train_output.parent.mkdir(parents=True, exist_ok=True)
    test_output.parent.mkdir(parents=True, exist_ok=True)

    train_df.write_csv(train_output)
    test_df.write_csv(test_output)


def main(
    reviews_path: Path = typer.Option(
        Path("data/preprocessed/preprocessed_reviews.csv"), "--reviews-path"
    ),
    train_path: Path = typer.Option(
        Path("data/split/train_reviews.csv"), "--train-path"
    ),
    test_path: Path = typer.Option(Path("data/split/test_reviews.csv"), "--test-path"),
    test_size: float = typer.Option(0.2, "--test-size"),
    random_state: int = typer.Option(42, "--random-state"),
    stratify: str = typer.Option("false", "--stratify"),
) -> None:
    split_data(
        reviews_path=str(reviews_path),
        train_path=str(train_path),
        test_path=str(test_path),
        test_size=test_size,
        random_state=random_state,
        stratify=_str_to_bool(stratify),
    )


if __name__ == "__main__":
    typer.run(main)

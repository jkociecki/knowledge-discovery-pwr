from pathlib import Path
import ast
from typing import Mapping, Sequence

import polars as pl
import typer


class Preprocessor:
    TARGET_COLUMN = "LABEL-simple_rating"
    DEFAULT_COLUMNS_TO_DROP = ["Unnamed: 0", "author_id"]

    PRODUCTS_META_SCHEMA: Mapping[str, pl.DataType] = {
        "product_id": pl.Utf8,
        "product_name": pl.Utf8,
        "brand_id": pl.Utf8,
        "brand_name": pl.Utf8,
        "loves_count": pl.Int32,
        "rating": pl.Float32,
        "reviews": pl.Int32,
        "size": pl.Utf8,
        "variation_type": pl.Utf8,
        "variation_value": pl.Utf8,
        "variation_desc": pl.Utf8,
        "ingredients": pl.Utf8,
        "price_usd": pl.Float32,
        "value_price_usd": pl.Float32,
        "sale_price_usd": pl.Float32,
        "limited_edition": pl.Boolean,
        "new": pl.Boolean,
        "online_only": pl.Boolean,
        "out_of_stock": pl.Boolean,
        "sephora_exclusive": pl.Boolean,
        "highlights": pl.Utf8,
        "primary_category": pl.Utf8,
        "secondary_category": pl.Utf8,
        "tertiary_category": pl.Utf8,
        "child_count": pl.Int32,
        "child_max_price": pl.Float32,
        "child_min_price": pl.Float32,
    }

    REVIEWS_SCHEMA: Mapping[str, pl.DataType] = {
        "Unnamed: 0": pl.Utf8,
        "author_id": pl.Utf8,
        "LABEL-simple_rating": pl.Utf8,
        "is_recommended": pl.Boolean,
        "helpfulness": pl.Float32,
        "total_feedback_count": pl.Int32,
        "total_neg_feedback_count": pl.Int32,
        "total_pos_feedback_count": pl.Int32,
        "submission_time": pl.Datetime,
        "review_text": pl.Utf8,
        "review_title": pl.Utf8,
        "skin_tone": pl.Utf8,
        "eye_color": pl.Utf8,
        "skin_type": pl.Utf8,
        "hair_color": pl.Utf8,
        "product_id": pl.Utf8,
        "product_name": pl.Utf8,
        "brand_name": pl.Utf8,
        "price_usd": pl.Float32,
    }

    CSV_SCAN_OPTIONS = {
        "encoding": "utf8-lossy",
        "truncate_ragged_lines": True,
        "ignore_errors": True,
    }

    SIMPLE_RATING_MAP = {
        "1-2": 1,
        "3-4": 2,
        "5": 3,
    }

    def __init__(
        self,
        products_metadata: Path,
        reviews_batches: Sequence[Path],
        columns_to_drop: Sequence[str] | None = None,
    ):
        self.products_metadata = Path(products_metadata)
        self.reviews_batches = [Path(path) for path in reviews_batches]
        self.columns_to_drop = (
            list(columns_to_drop)
            if columns_to_drop is not None
            else list(self.DEFAULT_COLUMNS_TO_DROP)
        )

    @staticmethod
    def parse_columns_to_drop(columns_to_drop: str | None) -> list[str]:
        if columns_to_drop is None:
            return list(Preprocessor.DEFAULT_COLUMNS_TO_DROP)

        parsed_value = columns_to_drop.strip()
        if not parsed_value:
            return []

        if parsed_value.startswith("[") and parsed_value.endswith("]"):
            loaded = ast.literal_eval(parsed_value)
            if not isinstance(loaded, list):
                raise ValueError("columns_to_drop list must be a list of strings")
            return [str(column).strip() for column in loaded if str(column).strip()]

        return [column.strip() for column in parsed_value.split(",") if column.strip()]

    def _load_reviews_lazy(self) -> pl.LazyFrame:
        expected_review_columns = list(self.REVIEWS_SCHEMA.keys())
        review_frames = [
            pl.scan_csv(
                batch_path,
                schema_overrides=self.REVIEWS_SCHEMA,
                **self.CSV_SCAN_OPTIONS,
            ).select(expected_review_columns)
            for batch_path in self.reviews_batches
        ]
        return pl.concat(review_frames, how="vertical")

    @classmethod
    def _normalize_target(cls, frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame.with_columns(
            pl.col(cls.TARGET_COLUMN)
            .replace_strict(cls.SIMPLE_RATING_MAP, default=None)
            .cast(pl.Int32)
            .alias(cls.TARGET_COLUMN)
        )

    def run_preprocessing_lazy(self) -> pl.LazyFrame:
        products_meta_lf = pl.scan_csv(
            self.products_metadata,
            schema_overrides=self.PRODUCTS_META_SCHEMA,
            **self.CSV_SCAN_OPTIONS,
        )

        return (
            self._normalize_target(self._load_reviews_lazy())
            .join(products_meta_lf, on="product_id", how="left")
            .drop(self.columns_to_drop)
            .unique()
        )

    def run_preprocessing(self) -> pl.DataFrame:
        return self.run_preprocessing_lazy().collect(engine="streaming")

    def preprocess_and_save(
        self,
        output_path: Path,
    ) -> Path:
        """Run preprocessing and save one final CSV output file."""
        output_path = Path(output_path)
        final_output = (
            output_path
            if output_path.suffix == ".csv"
            else output_path.with_suffix(".csv")
        )
        final_output.parent.mkdir(parents=True, exist_ok=True)
        result_df = self.run_preprocessing()
        result_df.write_csv(final_output)
        return final_output


def main(
    raw_dir: Path = typer.Option(Path("data/raw"), "--raw-dir"),
    reviews_pattern: str = typer.Option('"reviews_*.csv"', "--reviews-pattern"),
    products_filename: str = typer.Option("product_info.csv", "--products-filename"),
    columns_to_drop: str | None = typer.Option(None, "--columns-to-drop"),
    output_path: Path = typer.Option(
        Path("data/preprocessed/preprocessed_reviews.csv"), "--output-path"
    ),
) -> None:
    raw_data_dir = raw_dir
    reviews_batches = sorted(raw_data_dir.glob(reviews_pattern))

    if not reviews_batches:
        raise FileNotFoundError(
            f"No review files found in {raw_data_dir} matching pattern {reviews_pattern}"
        )

    preprocessor = Preprocessor(
        products_metadata=raw_data_dir / products_filename,
        reviews_batches=reviews_batches,
        columns_to_drop=Preprocessor.parse_columns_to_drop(columns_to_drop),
    )

    saved_file = preprocessor.preprocess_and_save(
        output_path=output_path,
    )
    print(saved_file)


if __name__ == "__main__":
    typer.run(main)

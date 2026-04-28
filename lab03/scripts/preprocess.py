from pathlib import Path
from typing import Mapping, Sequence

import polars as pl
import typer
import yaml


class Preprocessor:
    TARGET_COLUMN = "LABEL-simple_rating"

    DEFAULT_COLUMNS_TO_DROP = ["Unnamed: 0", "author_id"]
    DEFAULT_FILL_WITH_UNKNOWN_COLUMNS: list[str] = []

    BINARY_FLAG_COLUMNS = [
        "limited_edition",
        "new",
        "online_only",
        "out_of_stock",
        "sephora_exclusive",
    ]

    PRODUCTS_META_SCHEMA: Mapping[str, pl.DataType] = {
        "product_id": pl.Utf8,
        "product_name": pl.Utf8,
        "brand_id": pl.Utf8,
        "brand_name": pl.Utf8,
        "loves_count": pl.Int32,
        "rating": pl.Float32,
        "reviews": pl.Float32,
        "size": pl.Utf8,
        "variation_type": pl.Utf8,
        "variation_value": pl.Utf8,
        "variation_desc": pl.Utf8,
        "ingredients": pl.Utf8,
        "price_usd": pl.Float32,
        "value_price_usd": pl.Float32,
        "sale_price_usd": pl.Float32,
        "limited_edition": pl.Int8,
        "new": pl.Int8,
        "online_only": pl.Int8,
        "out_of_stock": pl.Int8,
        "sephora_exclusive": pl.Int8,
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
        "is_recommended": pl.Float32,
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
        fill_with_unknown_columns: Sequence[str] | None = None,
    ):
        self.products_metadata = Path(products_metadata)
        self.reviews_batches = [Path(path) for path in reviews_batches]
        self.columns_to_drop = (
            list(columns_to_drop)
            if columns_to_drop is not None
            else list(self.DEFAULT_COLUMNS_TO_DROP)
        )
        self.fill_with_unknown_columns = (
            list(fill_with_unknown_columns)
            if fill_with_unknown_columns is not None
            else list(self.DEFAULT_FILL_WITH_UNKNOWN_COLUMNS)
        )

    @staticmethod
    def _load_config(config_path: Path | None) -> dict:
        if config_path is None:
            return {}
        return yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}

    @staticmethod
    def load_columns_to_drop(config_path: Path | None) -> list[str]:
        if config_path is None:
            return list(Preprocessor.DEFAULT_COLUMNS_TO_DROP)

        config = Preprocessor._load_config(config_path)
        columns_to_drop = config.get(
            "columns_to_drop", Preprocessor.DEFAULT_COLUMNS_TO_DROP
        )

        if not isinstance(columns_to_drop, list):
            raise ValueError("'columns_to_drop' must be a YAML list")

        return [
            str(column).strip() for column in columns_to_drop if str(column).strip()
        ]

    @staticmethod
    def load_fill_with_unknown_columns(config_path: Path | None) -> list[str]:
        if config_path is None:
            return list(Preprocessor.DEFAULT_FILL_WITH_UNKNOWN_COLUMNS)

        config = Preprocessor._load_config(config_path)
        fill_with_unknown = config.get(
            "fill_with_unknown", Preprocessor.DEFAULT_FILL_WITH_UNKNOWN_COLUMNS
        )

        if not isinstance(fill_with_unknown, list):
            raise ValueError("'fill_with_unknown' must be a YAML list")

        return [
            str(column).strip() for column in fill_with_unknown if str(column).strip()
        ]

    def _fill_with_unknown(self, frame: pl.LazyFrame) -> pl.LazyFrame:
        if not self.fill_with_unknown_columns:
            return frame

        available_columns = set(frame.collect_schema().names())
        missing_columns = sorted(
            set(self.fill_with_unknown_columns).difference(available_columns)
        )
        if missing_columns:
            raise ValueError(
                f"Columns from 'fill_with_unknown' not found in dataset: {missing_columns}"
            )

        return frame.with_columns(
            [
                pl.col(column).cast(pl.Utf8).fill_null("unknown").alias(column)
                for column in self.fill_with_unknown_columns
            ]
        )

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

    @classmethod
    def _normalize_product_flags(cls, frame: pl.LazyFrame) -> pl.LazyFrame:
        return frame.with_columns(
            [
                pl.col(column).cast(pl.Boolean).alias(column)
                for column in cls.BINARY_FLAG_COLUMNS
            ]
        )

    @staticmethod
    def _add_text_features(frame: pl.LazyFrame) -> pl.LazyFrame:
        text = pl.col("review_text").fill_null("")
        char_len = text.str.len_chars().alias("char_len")
        word_len_raw = text.str.count_matches(r"\b\w+\b")
        letters_count = text.str.count_matches(r"[A-Za-z]")
        special_count = text.str.count_matches(r"[^\w\s]")
        digit_count = text.str.count_matches(r"\d")
        exclamations_count = text.str.count_matches(r"!")

        return frame.with_columns(
            [
                char_len,
                word_len_raw.alias("word_len"),
                (
                    letters_count.cast(pl.Float32)
                    / pl.when(word_len_raw > 0).then(word_len_raw).otherwise(1)
                ).alias("avg_word_len"),
                (
                    special_count.cast(pl.Float32)
                    / pl.when(char_len > 0).then(char_len).otherwise(1)
                ).alias("special_ratio"),
                (
                    digit_count.cast(pl.Float32)
                    / pl.when(char_len > 0).then(char_len).otherwise(1)
                ).alias("digit_ratio"),
                exclamations_count.alias("exclamations"),
                (exclamations_count > 0).cast(pl.Int8).alias("has_exclamation"),
            ]
        )

    def run_preprocessing_lazy(self) -> pl.LazyFrame:
        products_meta_lf = pl.scan_csv(
            self.products_metadata,
            schema_overrides=self.PRODUCTS_META_SCHEMA,
            **self.CSV_SCAN_OPTIONS,
        )
        products_meta_lf = self._normalize_product_flags(products_meta_lf)

        return (
            self._normalize_target(self._load_reviews_lazy())
            .join(products_meta_lf, on="product_id", how="left")
            .pipe(self._fill_with_unknown)
            .pipe(self._add_text_features)
            .drop(self.columns_to_drop)
            .drop_nulls()
            .unique()
        )

    def run_preprocessing(self) -> pl.DataFrame:
        return self.run_preprocessing_lazy().collect(engine="streaming")

    def preprocess_and_save(
        self,
        output_path: Path,
    ) -> Path:

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
    preprocess_config: Path = typer.Option(
        Path("preprocess_config.yaml"), "--preprocess-config"
    ),
    output_path: Path = typer.Option(
        Path("data/preprocessed/preprocessed_reviews.csv"), "--output-path"
    ),
) -> None:

    raw_data_dir = raw_dir
    reviews_batches = sorted(raw_data_dir.glob(reviews_pattern))

    preprocessor = Preprocessor(
        products_metadata=raw_data_dir / products_filename,
        reviews_batches=reviews_batches,
        columns_to_drop=Preprocessor.load_columns_to_drop(preprocess_config),
        fill_with_unknown_columns=Preprocessor.load_fill_with_unknown_columns(
            preprocess_config
        ),
    )

    _ = preprocessor.preprocess_and_save(
        output_path=output_path,
    )


if __name__ == "__main__":
    typer.run(main)

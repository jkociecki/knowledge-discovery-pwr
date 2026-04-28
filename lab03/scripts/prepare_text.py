import importlib
from pathlib import Path

import polars as pl
import typer
from tqdm import tqdm

TARGET_COLUMN = "LABEL-simple_rating"
TEXT_COLUMN = "review_text"
CLEAN_TEXT_COLUMN = "review_text_clean"


def _load_csv(path: Path) -> pl.DataFrame:
    return pl.read_csv(
        path,
        schema_overrides={TARGET_COLUMN: pl.Int32},
        encoding="utf8-lossy",
        truncate_ragged_lines=True,
        ignore_errors=True,
    )


def _load_nlp(language_model: str):
    spacy = importlib.import_module("spacy")
    try:
        return spacy.load(language_model, disable=["ner", "parser", "textcat"])
    except OSError:
        print(
            f"Language model '{language_model}' not found. Please install it using 'python -m spacy download {language_model}' and try again."
        )
        return spacy.blank("en")


def _clean_texts(texts: list[str], language_model: str, batch_size: int) -> list[str]:
    nlp = _load_nlp(language_model)
    cleaned: list[str] = []

    pipe = nlp.pipe(texts, batch_size=batch_size)
    for doc in tqdm(
        pipe, total=len(texts), desc="spaCy lemmatization", unit="doc", mininterval=1.0
    ):
        tokens: list[str] = []
        for token in doc:
            if not token.is_alpha or token.is_stop:
                continue
            lemma = token.lemma_.lower().strip() if token.lemma_ else ""
            if not lemma or lemma == "-pron-":
                lemma = token.text.lower().strip()
            if lemma:
                tokens.append(lemma)

        cleaned.append(" ".join(tokens) if tokens else "emptyreview")

    return cleaned


def prepare_text(
    input_path: Path,
    output_path: Path,
    language_model: str = "en_core_web_sm",
    batch_size: int = 256,
) -> Path:
    df = _load_csv(input_path)

    texts = df.get_column(TEXT_COLUMN).fill_null("").cast(pl.Utf8).to_list()

    cleaned_texts = _clean_texts(
        texts, language_model=language_model, batch_size=batch_size
    )

    result = df.with_columns(pl.Series(name=CLEAN_TEXT_COLUMN, values=cleaned_texts))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_csv(output_path)
    return output_path


def main(
    input_path: Path = typer.Option(
        Path("data/preprocessed/preprocessed_reviews.csv"), "--input-path"
    ),
    output_path: Path = typer.Option(
        Path("data/preprocessed/preprocessed_reviews_text_clean.csv"), "--output-path"
    ),
    language_model: str = typer.Option("en_core_web_sm", "--language-model"),
    batch_size: int = typer.Option(256, "--batch-size"),
) -> None:
    out = prepare_text(
        input_path=input_path,
        output_path=output_path,
        language_model=language_model,
        batch_size=batch_size,
    )
    print(f"Saved cleaned dataset to: {out}")


if __name__ == "__main__":
    typer.run(main)

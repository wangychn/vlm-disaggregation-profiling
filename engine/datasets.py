"""Helpers for loading the MME benchmark through Hugging Face datasets.

The probe script currently expects image paths, so this module also materializes
only the selected MME images to a local directory and yields their paths along
with the corresponding question text.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Optional

MME_DATASET_ID = "lmms-lab/MME"
DEFAULT_SPLIT = "test"


def _cache_dir_value(cache_dir: str | Path | None) -> str | None:
    if cache_dir is None:
        return None
    return str(Path(cache_dir).expanduser())


def _split_spec(split: str, max_items: int | None, streaming: bool) -> str:
    if streaming or max_items is None:
        return split
    return f"{split}[:{max_items}]"


def load_mme_dataset(
    *,
    split: str = DEFAULT_SPLIT,
    cache_dir: str | Path | None = None,
    max_items: int | None = None,
    streaming: bool = False,
):
    """Load the MME dataset from Hugging Face.

    Args:
        split: dataset split, usually ``"test"`` for MME.
        cache_dir: optional Hugging Face datasets cache directory.
        max_items: load only the first N rows when not streaming.
        streaming: if True, stream rows lazily instead of preparing the full
            requested slice in the local datasets cache.
    """
    from datasets import load_dataset

    return load_dataset(
        MME_DATASET_ID,
        split=_split_spec(split, max_items=max_items, streaming=streaming),
        cache_dir=_cache_dir_value(cache_dir),
        streaming=streaming,
    )


def iter_mme_examples(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    cache_dir: str | Path | None = None,
    max_items: int | None = None,
    streaming: bool = False,
    category: str | None = None,
) -> Iterator[tuple[Path, str, dict[str, str]]]:
    """Yield local image paths and questions from MME.

    The underlying dataset is loaded from Hugging Face and cached automatically.
    The image payloads are then written under ``root/images/<split>/...`` only
    for the selected rows so the rest of the probe can keep using local paths.
    """
    root = Path(root)
    export_root = root / "images" / split
    export_root.mkdir(parents=True, exist_ok=True)

    dataset = load_mme_dataset(
        split=split,
        cache_dir=cache_dir,
        max_items=max_items if category is None else None,
        streaming=streaming,
    )

    yielded = 0
    for row in dataset:
        row_category = str(row.get("category", "uncategorized"))
        if category and row_category != category:
            continue

        image = row["image"]
        question = str(row["question"])
        answer = str(row.get("answer", ""))
        question_id = str(row.get("question_id", yielded))
        safe_id = question_id.replace("/", "_")

        suffix = ".png"
        if getattr(image, "format", None):
            suffix = "." + str(image.format).lower()

        image_dir = export_root / row_category
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / f"{safe_id}{suffix}"

        if not image_path.exists():
            image.save(image_path)

        yield image_path, question, {
            "question_id": question_id,
            "answer": answer,
            "category": row_category,
        }

        yielded += 1
        if max_items is not None and yielded >= max_items:
            break


def materialize_mme_examples(
    root: Path,
    *,
    split: str = DEFAULT_SPLIT,
    cache_dir: str | Path | None = None,
    max_items: int | None = None,
    streaming: bool = False,
    category: str | None = None,
) -> int:
    """Populate a local image cache for a subset of MME and return the count."""
    count = 0
    for _image_path, _question, _meta in iter_mme_examples(
        root,
        split=split,
        cache_dir=cache_dir,
        max_items=max_items,
        streaming=streaming,
        category=category,
    ):
        count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MME examples via Hugging Face datasets.")
    parser.add_argument("--root", type=Path, required=True, help="local export directory")
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="dataset split")
    parser.add_argument("--cache-dir", default=None, help="datasets cache directory")
    parser.add_argument("--max-items", type=int, default=None, help="only keep the first N items")
    parser.add_argument("--streaming", action="store_true", help="stream rows instead of preparing the full split")
    parser.add_argument("--category", default=None, help="optional single MME category filter")
    args = parser.parse_args()

    count = materialize_mme_examples(
        args.root,
        split=args.split,
        cache_dir=args.cache_dir,
        max_items=args.max_items,
        streaming=args.streaming,
        category=args.category,
    )
    print(f"Prepared {count} MME example(s) under {args.root}")

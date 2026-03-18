"""Utility helpers for obtaining common multimodal evaluation datasets.

The functions in this module are intended to centralize dataset fetching and
simple parsing so that the benchmarking/profiling scripts can iterate over a
set of (image, prompt) examples without caring about the underlying format.
Currently we provide a basic loader for COCO captions; additional datasets can
be added as needed.
"""

from pathlib import Path
from typing import Iterator, Tuple, Optional


def download_coco(dest: Path, split: str = "val2017") -> None:
    """Prepare disk paths for COCO and optionally download images.

    The COCO image files are large and normally obtained from the official
    website.  If the ``requests`` and ``tarfile`` modules are available the
    helper will fetch and extract them automatically; otherwise it just creates
    the expected directory layout and returns.
    """
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "annotations").mkdir(exist_ok=True)
    (dest / "images").mkdir(exist_ok=True)

    # try automatic download of the image and annotation archives
    try:
        import requests
        import zipfile
    except ImportError:
        return

    image_url = f"https://images.cocodataset.org/zips/{split}.zip"
    image_archive = dest / f"{split}.zip"
    if not image_archive.exists():
        r = requests.get(image_url, stream=True)
        r.raise_for_status()
        with open(image_archive, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)

    annotations_url = "https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annotations_archive = dest / "annotations_trainval2017.zip"
    if not annotations_archive.exists():
        r = requests.get(annotations_url, stream=True)
        r.raise_for_status()
        with open(annotations_archive, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)

    if not (dest / "images" / split).exists():
        with zipfile.ZipFile(image_archive, "r") as zf:
            zf.extractall(dest / "images")

    ann_file = dest / "annotations" / f"captions_{split}.json"
    if not ann_file.exists():
        with zipfile.ZipFile(annotations_archive, "r") as zf:
            zf.extractall(dest)


def load_coco_captions(
    root: Path, split: str = "val2017", max_items: Optional[int] = None
) -> Iterator[Tuple[Path, str]]:
    """Iterate over COCO caption examples.

    Returns pairs of ``(image_path, caption)``.  Stops early if ``max_items``
    is set.
    """
    from pycocotools.coco import COCO

    ann_file = root / "annotations" / f"captions_{split}.json"
    coco = COCO(str(ann_file))
    img_ids = coco.getImgIds()
    count = 0
    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        file_path = root / "images" / split / img_info["file_name"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            yield file_path, ann["caption"]
            count += 1
            if max_items and count >= max_items:
                return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Helper for COCO dataset")
    parser.add_argument("--root", type=Path, required=True, help="destination directory")
    parser.add_argument("--split", default="val2017", help="COCO split")
    parser.add_argument("--download", action="store_true", help="fetch images archive")
    args = parser.parse_args()

    if args.download:
        download_coco(args.root, split=args.split)
    else:
        print("Run with --download to fetch images and prepare folders.")

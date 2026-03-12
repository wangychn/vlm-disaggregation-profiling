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
    """Prepare disk paths for COCO data.

    Creates ``dest/annotations`` and ``dest/images`` if missing.  The user
    still needs to fetch the official tarballs and unpack them in place.
    """
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "annotations").mkdir(exist_ok=True)
    (dest / "images").mkdir(exist_ok=True)


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
        file_path = root / "images" / img_info["file_name"]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            yield file_path, ann["caption"]
            count += 1
            if max_items and count >= max_items:
                return

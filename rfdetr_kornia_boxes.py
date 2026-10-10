"""Exercise RF-DETR's actual batch hook with synthetic continuous box edges."""

import argparse
import importlib.metadata
import json
from types import SimpleNamespace

import torch
from rfdetr.datasets.kornia_transforms import build_kornia_pipeline
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.utilities.tensors import NestedTensor
from torchvision.ops import box_convert


def probe(segmentation: bool) -> dict:
    height, width = 6, 10
    image = torch.zeros(1, 3, height, width)
    image[:, :, 1:4, 2:6] = 1
    masks = torch.stack((image[0, 0].bool(), torch.ones(height, width).bool()))
    target = {
        "boxes": torch.tensor([[2.0, 1.0, 6.0, 4.0], [0.0, 0.0, 10.0, 6.0]]),
        "labels": torch.tensor([1, 2]),
        "area": torch.tensor([12.0, 60.0]),
        "iscrowd": torch.zeros(2, dtype=torch.int64),
    }
    if segmentation:
        target["masks"] = masks
    module = SimpleNamespace(
        trainer=SimpleNamespace(training=True),
        model_config=SimpleNamespace(
            segmentation_head=segmentation, use_grouppose_keypoints=False
        ),
        train_config=SimpleNamespace(keypoint_flip_pairs=None),
        _kornia_pipeline=build_kornia_pipeline(
            {"HorizontalFlip": {"p": 1.0}}, width, with_masks=True
        ),
        _kornia_normalize=torch.nn.Identity(),
    )
    samples, targets = RFDETRDataModule.on_after_batch_transfer(
        module, (NestedTensor(image, None), [target]), 0
    )
    actual = box_convert(targets[0]["boxes"], "cxcywh", "xyxy") * torch.tensor(
        [width, height, width, height]
    )
    expected = torch.tensor([[4.0, 1.0, 8.0, 4.0], [0.0, 0.0, 10.0, 6.0]])
    assert torch.equal(samples.tensors, image.flip(-1))
    if segmentation:
        assert torch.equal(targets[0]["masks"].bool(), masks.flip(-1))
    return {
        "segmentation": segmentation,
        "expected_xyxy": expected.tolist(),
        "actual_xyxy": actual.tolist(),
        "max_box_error_pixels": float((actual - expected).abs().max()),
        "full_frame_area": float(targets[0]["area"][1]),
        "pixels_and_masks_correct": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", choices=("bug", "fixed"), default="bug")
    args = parser.parse_args()
    cases = [probe(False), probe(True)]
    print(
        json.dumps(
            {
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("rfdetr", "kornia", "torch")
                },
                "cases": cases,
            },
            indent=2,
        )
    )
    for case in cases:
        if args.expect == "fixed":
            assert case["max_box_error_pixels"] < 1e-5
            assert case["full_frame_area"] == 60
        else:
            assert abs(case["max_box_error_pixels"] - 1) < 1e-5
            assert case["full_frame_area"] == 54


if __name__ == "__main__":
    main()

"""Exercise released adapters with synthetic pixels and boxes, without weights."""

import inspect
import io
import json
from contextlib import redirect_stdout
import albumentations as A
import numpy as np
from PIL import Image


def check_integrations():
    import supervision as sv
    import torch
    from rfdetr.datasets.transforms import AlbumentationsWrapper
    from ultralytics.data.augment import Albumentations
    from ultralytics.utils.instance import Instances

    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    boxes = np.array([[1, 1, 3, 3]], dtype=np.float32)
    labels = np.array([2])
    expected = np.array([[4, 1, 6, 3]], dtype=np.float32)
    format_key = (
        "coord_format"
        if "coord_format" in inspect.signature(A.BboxParams).parameters
        else "format"
    )
    transform = A.Compose(
        [A.HorizontalFlip(p=1)],
        bbox_params=A.BboxParams(**{format_key: "pascal_voc"}, label_fields=["labels"]),
    )
    result = transform(image=image, bboxes=boxes, labels=labels)
    detections = sv.Detections(
        xyxy=np.asarray(result["bboxes"]),
        class_id=np.asarray(result["labels"], dtype=int),
    )
    np.testing.assert_allclose(detections.xyxy, expected, atol=1e-5)
    np.testing.assert_array_equal(detections.class_id, labels)
    report = {"supervision_boxes": "passed"}

    target = {"boxes": torch.tensor(boxes), "labels": torch.tensor(labels)}
    pixel = AlbumentationsWrapper(A.InvertImg(p=1))
    output, _ = pixel(Image.fromarray(image), target)
    np.testing.assert_array_equal(np.asarray(output), 255 - image)
    report["rfdetr_pixel"] = "passed"
    try:
        spatial = AlbumentationsWrapper(A.HorizontalFlip(p=1))
        _, updated = spatial(Image.fromarray(image), target)
        np.testing.assert_allclose(updated["boxes"].numpy(), expected, atol=1e-5)
        report["rfdetr_spatial"] = "passed"
    except TypeError as error:
        report["rfdetr_spatial"] = f"incompatible: {error}"

    def sample():
        return {
            "img": image.copy(),
            "cls": labels.reshape(-1, 1).copy(),
            "instances": Instances(
                boxes.copy(),
                np.empty((0, 1000, 2), dtype=np.float32),
                bbox_format="xyxy",
                normalized=False,
            ),
        }

    default = Albumentations(p=1)
    assert default.transform is not None, "default YOLO adapter silently disabled"
    for _ in range(20):
        result = default(sample())
        assert result["img"].shape == image.shape
        np.testing.assert_array_equal(result["cls"], labels.reshape(-1, 1))
    report["ultralytics_default"] = "passed"
    pixel = Albumentations(p=1, transforms=[A.InvertImg(p=1)])
    assert pixel.transform is not None, "pixel YOLO adapter silently disabled"
    np.testing.assert_array_equal(pixel(sample())["img"], 255 - image)
    report["ultralytics_pixel"] = "passed"
    spatial = Albumentations(p=1, transforms=[A.HorizontalFlip(p=1)])
    if spatial.transform is None:
        report["ultralytics_spatial"] = (
            "incompatible: adapter caught initialization error and disabled augmentation"
        )
    else:
        result = spatial(sample())
        instances = result["instances"]
        instances.convert_bbox("xyxy")
        instances.denormalize(7, 5)
        np.testing.assert_allclose(instances.bboxes, expected, atol=1e-5)
        report["ultralytics_spatial"] = "passed"
    return report


if __name__ == "__main__":
    with redirect_stdout(io.StringIO()):
        result = check_integrations()
    print(json.dumps(result, indent=2))

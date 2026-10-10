"""Synthetic probe of seeded flip decisions; no models or datasets required."""

import argparse
import hashlib
import json
from collections import Counter
from importlib import metadata
import albumentations as A
import numpy as np


def pipeline(seed):
    return A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)], seed=seed)


def sequence(transform, calls):
    image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    variants = (image, image[:, ::-1], image[::-1], image[::-1, ::-1])
    outcomes = []
    for _ in range(calls):
        output = transform(image=image)["image"]
        matches = [
            i for i, variant in enumerate(variants) if np.array_equal(output, variant)
        ]
        assert len(matches) == 1, "output must match exactly one orientation"
        outcomes.append(matches[0])
    return outcomes


def check(seed, calls):
    transform = pipeline(seed)
    first = sequence(transform, calls)
    repeated = sequence(pipeline(seed), calls)
    transform.set_random_seed(seed)
    reseeded = sequence(transform, calls)
    counts = Counter(first)
    names = ("neither", "horizontal_only", "vertical_only", "both")
    probabilities = [counts[i] / calls for i in range(4)]
    p_h = probabilities[1] + probabilities[3]
    p_v = probabilities[2] + probabilities[3]
    variance = p_h * (1 - p_h) * p_v * (1 - p_v)
    phi = (probabilities[3] - p_h * p_v) / variance**0.5 if variance else None
    return {
        "seed": seed,
        "calls": calls,
        "counts": {name: counts[i] for i, name in enumerate(names)},
        "repeatable": first == repeated,
        "reseed_repeatable": first == reseeded,
        "sequence_sha256": hashlib.sha256(bytes(first)).hexdigest(),
        "phi_correlation": phi,
    }


def report(calls=400, seeds=(0, 1, 42, 123)):
    distributions = {}
    for name in (
        "albumentations",
        "albumentationsx",
        "albucore",
        "torch",
        "numpy",
        "opencv-python-headless",
        "opencv-python",
    ):
        try:
            distributions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    assert not {"albumentations", "albumentationsx"} <= distributions.keys(), (
        "overlapping distributions installed"
    )
    return {
        "distributions": distributions,
        "import_version": A.__version__,
        "results": [check(seed, calls) for seed in seeds],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 42, 123])
    args = parser.parse_args()
    if args.calls <= 0:
        parser.error("--calls must be positive")
    print(json.dumps(report(args.calls, args.seeds), indent=2))

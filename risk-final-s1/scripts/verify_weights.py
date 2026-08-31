#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata" / "weights_manifest.tsv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify Final-S1 model weights against "
            "the frozen release manifest."
        )
    )

    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="Directory containing the 20 Final-S1 .safetensors files.",
    )

    args = parser.parse_args()

    weights_dir = args.weights_dir.resolve()

    with MANIFEST.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(
                handle,
                delimiter="\t",
            )
        )

    if len(rows) != 20:
        raise RuntimeError(
            f"Expected 20 manifest rows; found {len(rows)}"
        )

    failed = 0

    for index, row in enumerate(rows, start=1):
        filename = row["filename"]
        expected_bytes = int(row["bytes"])
        expected_sha256 = row["sha256"]

        path = weights_dir / filename

        if not path.is_file():
            print(
                f"[{index:02d}/20] MISSING  {filename}"
            )
            failed += 1
            continue

        actual_bytes = path.stat().st_size

        if actual_bytes != expected_bytes:
            print(
                f"[{index:02d}/20] SIZE_FAIL  "
                f"{filename} "
                f"expected={expected_bytes} "
                f"actual={actual_bytes}"
            )
            failed += 1
            continue

        actual_sha256 = sha256_file(path)

        if actual_sha256 != expected_sha256:
            print(
                f"[{index:02d}/20] SHA256_FAIL  "
                f"{filename}"
            )
            failed += 1
            continue

        print(
            f"[{index:02d}/20] PASS  {filename}"
        )

    if failed:
        print()
        print(
            f"WEIGHT_VERIFICATION=FAIL failed={failed}"
        )
        return 1

    print()
    print("WEIGHT_VERIFICATION=PASS")
    print("WEIGHT_COUNT=20")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

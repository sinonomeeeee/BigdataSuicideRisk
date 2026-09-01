#!/usr/bin/env python3

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent
RUNTIME = HERE / "runtime"
MANIFEST = HERE / "weights.sha256"

RUNTIME_SHA256 = {
    "ml384": {
        "config.json":
            "2999d67ceac7df103fe81064f5e9ff5034923b67a86478d6d03ef25ea0887eb4",
        "tokenizer_config.json":
            "0dbe41323167ba5f2e1e876ccd05068318dd2378b922bdfd924a389dba3bcc7a",
        "tokenizer.json":
            "849d4f376376b7023111b5965ebba54a11a552a0798508567087a3a475d148a0",
    },
    "ml640": {
        "config.json":
            "2999d67ceac7df103fe81064f5e9ff5034923b67a86478d6d03ef25ea0887eb4",
        "tokenizer_config.json":
            "56f16a46b18f056e6e6e5a9192b44caf890b1d5f10a726ec1107e8014770472d",
        "tokenizer.json":
            "7b823cd45836a906f3e470d9bff052759ec5fd6308271d01e2b26b4e51882496",
    },
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(16 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def specs():
    rows = []

    for seed in (514, 9, 42):
        for fold in range(5):
            rows.append(
                (
                    f"ml384_seed{seed}_fold{fold}.safetensors",
                    "ml384",
                    seed,
                    fold,
                )
            )

    for fold in range(5):
        rows.append(
            (
                f"ml640_seed514_fold{fold}.safetensors",
                "ml640",
                514,
                fold,
            )
        )

    return rows


def read_manifest():
    result = {}

    with MANIFEST.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            digest, filename = line.split(None, 1)
            result[filename.strip()] = digest

    return result


def verify_runtime():
    for branch, files in RUNTIME_SHA256.items():
        for filename, expected in files.items():

            path = RUNTIME / branch / filename

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing runtime file: {path}"
                )

            actual = sha256_file(path)

            if actual != expected:
                raise RuntimeError(
                    f"Runtime SHA256 mismatch: {path}\n"
                    f"expected={expected}\n"
                    f"actual={actual}"
                )


def place_weight(source, destination, mode, force):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists() or destination.is_symlink():

        if not force:
            raise FileExistsError(
                f"Destination already exists: {destination}"
            )

        if destination.is_dir():
            raise RuntimeError(
                f"Refusing to replace directory: {destination}"
            )

        destination.unlink()

    if mode == "copy":
        shutil.copy2(source, destination)

    elif mode == "hardlink":
        os.link(source, destination)

    elif mode == "symlink":
        destination.symlink_to(source.resolve())

    else:
        raise ValueError(mode)


def copy_metadata(branch, fold_dir):
    for filename in (
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ):
        shutil.copy2(
            RUNTIME / branch / filename,
            fold_dir / filename,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the 20 Final-S1 Risk Hugging Face "
            "checkpoint directories from the flat Release assets."
        )
    )

    parser.add_argument(
        "--weights-dir",
        required=True,
        type=Path,
        help="Directory containing the 20 downloaded safetensors files.",
    )

    parser.add_argument(
        "--model-root",
        required=True,
        type=Path,
        help="Destination root for the prepared model directories.",
    )

    parser.add_argument(
        "--mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How model.safetensors is placed in each fold directory.",
    )

    parser.add_argument(
        "--skip-weight-hash",
        action="store_true",
        help="Skip SHA256 checking of the 20 model files.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing model.safetensors files.",
    )

    args = parser.parse_args()

    weights_dir = args.weights_dir.resolve()
    model_root = args.model_root.resolve()

    if not weights_dir.is_dir():
        raise FileNotFoundError(
            f"Weights directory not found: {weights_dir}"
        )

    verify_runtime()

    checkpoint_specs = specs()

    if len(checkpoint_specs) != 20:
        raise RuntimeError(
            "Internal checkpoint specification must contain 20 models."
        )

    manifest = read_manifest()

    missing = []

    for filename, _, _, _ in checkpoint_specs:
        if not (weights_dir / filename).is_file():
            missing.append(filename)

    if missing:
        raise FileNotFoundError(
            "Missing Release assets:\n"
            + "\n".join(missing)
        )

    print("[prepare] found 20 model weights")

    if not args.skip_weight_hash:

        print("[prepare] verifying model SHA256")

        for filename, _, _, _ in checkpoint_specs:

            source = weights_dir / filename
            expected = manifest.get(filename)

            if expected is None:
                raise RuntimeError(
                    f"No manifest entry for: {filename}"
                )

            actual = sha256_file(source)

            if actual != expected:
                raise RuntimeError(
                    f"Weight SHA256 mismatch: {filename}\n"
                    f"expected={expected}\n"
                    f"actual={actual}"
                )

        print("[prepare] weight SHA256 verification PASS")

    for filename, branch, seed, fold in checkpoint_specs:

        fold_dir = (
            model_root
            / branch
            / f"seed{seed}"
            / f"fold{fold}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        place_weight(
            weights_dir / filename,
            fold_dir / "model.safetensors",
            args.mode,
            args.force,
        )

        copy_metadata(
            branch,
            fold_dir,
        )

    prepared = list(
        model_root.glob(
            "ml*/seed*/fold*/model.safetensors"
        )
    )

    if len(prepared) != 20:
        raise RuntimeError(
            f"Expected 20 prepared models, found {len(prepared)}"
        )

    print("[prepare] prepared checkpoints = 20")
    print(f"[prepare] model root = {model_root}")
    print("PREPARE_MODELS=PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            f"PREPARE_MODELS=FAIL: {exc}",
            file=sys.stderr,
        )
        raise

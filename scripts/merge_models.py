"""CLI entry point for merging .safetensors checkpoints."""

from __future__ import annotations

import argparse
from typing import List

from modules.model_merger import merge_models


def parse_weights(raw: str) -> List[float]:
    return [float(part) for part in raw.split(",")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="Path for the merged .safetensors file")
    parser.add_argument(
        "--models",
        required=True,
        help="Comma separated list of model paths to merge",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Comma separated list of weights matching the provided models",
    )
    args = parser.parse_args()

    models = [part for part in args.models.split(",") if part]
    weights = parse_weights(args.weights)
    output = merge_models(models, weights, args.output)
    print(f"Merged model written to {output}")


if __name__ == "__main__":
    main()

"""CLI helper to download a Civitai model file."""

from __future__ import annotations

import argparse

from modules import civitai_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Model ID or Civitai model URL")
    parser.add_argument("--output-dir", dest="output_dir", help="Optional download directory")
    parser.add_argument("--filename", help="Optional filename for the downloaded model")
    parser.add_argument(
        "--config",
        help="Optional path to a civitai.yaml style config file overriding environment variables",
    )
    args = parser.parse_args()

    config = civitai_client.CivitaiConfig.from_env(args.config)
    path = civitai_client.download_model(
        args.model,
        destination_dir=args.output_dir,
        filename=args.filename,
        config=config,
    )
    print(f"Downloaded model to {path}")


if __name__ == "__main__":
    main()

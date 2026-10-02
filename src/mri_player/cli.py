import argparse
import sys
from pathlib import Path

from mri_player.config import load_config
from mri_player.converter import convert


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline WAV-to-gradient sequence converter for GE Discovery MR750 3.0T"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    convert_command = commands.add_parser("convert", help="Convert PCM WAV to Pulseq segments")
    convert_command.add_argument("input", type=Path, help="Input PCM WAV file")
    convert_command.add_argument("--config", required=True, type=Path, help="MR750 TOML config")
    convert_command.add_argument(
        "--output-dir", required=True, type=Path, help="New output directory"
    )
    arguments = parser.parse_args()

    try:
        config = load_config(arguments.config)
        manifest = convert(arguments.input, config, arguments.output_dir)
    except ValueError as error:
        parser.exit(2, f"Error: {error}\n")
    except OSError as error:
        parser.exit(2, f"File error: {error}\n")

    print(f"Generated {len(manifest['segments'])} Pulseq segments in {arguments.output_dir}")
    print("For offline review only. MR750 scanner compatibility has not been verified.")


if __name__ == "__main__":
    sys.exit(main())

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mri_player.audio import AudioProcessor, measure_waveform
from mri_player.config import PUBLIC_MAX_GRADIENT_MTM, PUBLIC_MAX_SLEW_TMS, MR750Config
from mri_player.sequence import write_sequence

SILENCE_PEAK = 1e-4


@dataclass(frozen=True)
class Measurement:
    peak: float
    slew_per_second: float


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def measure_song(processor: AudioProcessor) -> Measurement:
    peak = 0.0
    slew = 0.0
    for start, end in processor.segment_bounds():
        waveform = processor.read_segment(start, end)
        segment_peak, segment_slew = measure_waveform(waveform, processor.config.raster_seconds)
        peak = max(peak, segment_peak)
        slew = max(slew, segment_slew)
    return Measurement(peak, slew)


def choose_scale(measurement: Measurement, config: MR750Config) -> float:
    if measurement.peak < SILENCE_PEAK:
        return 0.0
    scale = min(config.target_peak_mtm, config.max_gradient_mtm * 0.99) / measurement.peak
    if measurement.slew_per_second > 0:
        scale = min(scale, config.max_slew_tms * 0.99 * 1000 / measurement.slew_per_second)
    return scale


def convert(input_path: Path, config: MR750Config, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError(f"Output directory already exists: {output_dir}")
    processor = AudioProcessor(input_path, config)
    measurement = measure_song(processor)
    scale = choose_scale(measurement, config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scanner_model": config.scanner_model,
        "offline_only": True,
        "profile": config.as_dict(),
        "public_nominal_hardware_limits": {
            "gradient_mtm_per_axis": PUBLIC_MAX_GRADIENT_MTM,
            "slew_tms_per_axis": PUBLIC_MAX_SLEW_TMS,
        },
        "input": {
            "filename": input_path.name,
            "sha256": hash_file(input_path),
            "sample_rate_hz": processor.info.sample_rate,
            "frames": processor.info.frames,
            "channels": processor.info.channels,
            "subtype": processor.info.subtype,
            "duration_seconds": processor.info.duration_seconds,
        },
        "channel_mapping": "mono_to_x" if processor.info.channels == 1 else "stereo_average_to_x",
        "segment_pause_note": "Each .seq file is executed separately; gapless playback has not been verified.",
        "global_scale_mtm_per_audio_unit": scale,
        "global_attenuation_db_from_target": (
            20 * math.log10(measurement.peak * scale / config.target_peak_mtm)
            if scale > 0
            else None
        ),
        "segments": [],
    }

    with tempfile.TemporaryDirectory(prefix="mri-player-", dir=output_dir.parent) as temp_name:
        temp_dir = Path(temp_name)
        for index, (start, end) in enumerate(processor.segment_bounds(), start=1):
            waveform = (
                processor.read_segment(start, end) * scale
                if scale > 0
                else np.zeros(end - start, dtype=np.float64)
            )
            peak, slew_per_second = measure_waveform(waveform, config.raster_seconds)
            slew_tms = slew_per_second / 1000
            if peak > config.max_gradient_mtm * (1 + 1e-9):
                raise ValueError(f"Segment {index} exceeds the gradient amplitude limit")
            if slew_tms > config.max_slew_tms * (1 + 1e-9):
                raise ValueError(f"Segment {index} exceeds the slew rate limit")
            filename = f"segment-{index:04d}.seq"
            sequence_path = temp_dir / filename
            metrics = write_sequence(waveform, config, sequence_path)
            manifest["segments"].append(
                {
                    "index": index,
                    "file": filename,
                    "source_start_seconds": start / config.output_sample_rate,
                    "source_end_seconds": min(
                        end / config.output_sample_rate, processor.info.duration_seconds
                    ),
                    "sequence_duration_seconds": len(waveform) * config.raster_seconds,
                    "samples": len(waveform),
                    "blocks": metrics.blocks,
                    "max_gradient_mtm": metrics.max_gradient_mtm,
                    "max_slew_tms": metrics.max_slew_tms,
                    "size_bytes": sequence_path.stat().st_size,
                    "sha256": hash_file(sequence_path),
                }
            )

        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temp_dir, output_dir)
    return manifest

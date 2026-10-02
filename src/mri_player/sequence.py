from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pypulseq as pp

from mri_player.config import MR750Config

BLOCK_SECONDS = 0.1


@dataclass(frozen=True)
class SequenceMetrics:
    blocks: int
    max_gradient_mtm: float
    max_slew_tms: float


def block_bounds(sample_count: int, block_samples: int) -> Iterator[tuple[int, int]]:
    start = 0
    while start < sample_count:
        end = min(start + block_samples, sample_count)
        if sample_count - end == 1:
            end = sample_count
        yield start, end
        start = end


def write_sequence(waveform_mtm: np.ndarray, config: MR750Config, path: Path) -> SequenceMetrics:
    system = pp.Opts(
        max_grad=config.max_gradient_mtm,
        grad_unit="mT/m",
        max_slew=config.max_slew_tms,
        slew_unit="T/m/s",
        grad_raster_time=config.raster_seconds,
        block_duration_raster=config.raster_seconds,
    )
    sequence = pp.Sequence(system)
    waveform = waveform_mtm * system.gamma * 1e-3
    block_samples = max(2, round(BLOCK_SECONDS / config.raster_seconds))
    count = 0

    for start, end in block_bounds(len(waveform), block_samples):
        if end - start < 2:
            raise ValueError("Gradient events require at least two samples")
        first = 0.0 if start == 0 else float((waveform[start - 1] + waveform[start]) / 2)
        last = 0.0 if end == len(waveform) else float((waveform[end - 1] + waveform[end]) / 2)
        gradient = pp.make_arbitrary_grad(
            channel="x",
            waveform=waveform[start:end],
            first=first,
            last=last,
            system=system,
        )
        sequence.add_block(gradient)
        count += 1

    timing_ok, timing_errors = sequence.check_timing()
    if not timing_ok:
        raise ValueError(f"Pulseq timing check failed: {timing_errors[:3]}")
    duration_seconds, _, _ = sequence.duration()
    expected_seconds = len(waveform) * config.raster_seconds
    if not np.isclose(duration_seconds, expected_seconds, atol=config.raster_seconds / 10):
        raise ValueError("Pulseq duration does not match the gradient samples")
    sequence.set_definition("Name", "mr750_gradient_audio")
    sequence.write(str(path))
    return validate_written_sequence(path, config, len(waveform), count)


def validate_written_sequence(
    path: Path, config: MR750Config, expected_samples: int, expected_blocks: int
) -> SequenceMetrics:
    sequence = pp.Sequence()
    sequence.read(str(path))
    timing_ok, timing_errors = sequence.check_timing()
    if not timing_ok:
        raise ValueError(f"Pulseq timing check failed after reading back: {timing_errors[:3]}")
    duration_seconds, block_count, _ = sequence.duration()
    if block_count != expected_blocks or not np.isclose(
        duration_seconds, expected_samples * config.raster_seconds, atol=config.raster_seconds / 10
    ):
        raise ValueError("Pulseq duration or block count changed after reading back")

    max_gradient = 0.0
    max_slew = 0.0
    previous_last = 0.0
    total_samples = 0
    for block_index in range(1, block_count + 1):
        block = sequence.get_block(block_index)
        if any(event is not None for event in (block.rf, block.adc, block.gy, block.gz)):
            raise ValueError("Unexpected non-X-gradient event after reading back")
        gradient = block.gx
        if gradient is None or not np.isclose(gradient.first, previous_last, atol=1e-3):
            raise ValueError("Gradient block boundary is discontinuous after reading back")
        values = gradient.waveform
        total_samples += len(values)
        max_gradient = max(
            max_gradient, float(np.max(np.abs(values))), abs(gradient.first), abs(gradient.last)
        )
        block_slew = max(
            float(np.max(np.abs(np.diff(values)))) / config.raster_seconds,
            2 * abs(float(values[0] - gradient.first)) / config.raster_seconds,
            2 * abs(float(gradient.last - values[-1])) / config.raster_seconds,
        )
        max_slew = max(max_slew, block_slew)
        previous_last = gradient.last
    if total_samples != expected_samples or not np.isclose(previous_last, 0, atol=1e-3):
        raise ValueError("Gradient sample count or endpoint changed after reading back")

    max_gradient_mtm = max_gradient / sequence.system.gamma * 1000
    max_slew_tms = max_slew / sequence.system.gamma
    if max_gradient_mtm > config.max_gradient_mtm * (1 + 1e-6):
        raise ValueError("Gradient amplitude exceeds the configured limit after reading back")
    if max_slew_tms > config.max_slew_tms * (1 + 1e-6):
        raise ValueError("Slew rate exceeds the configured limit after reading back")
    return SequenceMetrics(block_count, max_gradient_mtm, max_slew_tms)

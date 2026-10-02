import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfiltfilt

from mri_player.config import MR750Config

WINDOW_MARGIN_SECONDS = 0.1


def align_resampled_samples(
    resampled: np.ndarray,
    start_sample: int,
    end_sample: int,
    window_start: int,
    input_rate: int,
    output_rate: int,
) -> np.ndarray:
    sample_count = end_sample - start_sample
    # Integer phase avoids floating point drift in long audio
    phase_numerator = (2 * start_sample + 1) * input_rate - 2 * window_start * output_rate
    sample_offset, phase_remainder = divmod(phase_numerator, 2 * input_rate)
    if sample_offset < 0:
        raise ValueError("Resampling window does not cover the target sample time")

    result = np.empty(sample_count, dtype=np.float64)
    if phase_remainder == 0:
        available = min(sample_count, max(0, len(resampled) - sample_offset))
        result[:available] = resampled[sample_offset : sample_offset + available]
    elif phase_remainder == input_rate:
        available = min(sample_count, max(0, len(resampled) - sample_offset - 1))
        np.add(
            resampled[sample_offset : sample_offset + available],
            resampled[sample_offset + 1 : sample_offset + available + 1],
            out=result[:available],
        )
        result[:available] *= 0.5
    else:
        available = min(sample_count, max(0, len(resampled) - sample_offset - 1))
        phase = phase_remainder / (2 * input_rate)
        np.multiply(
            resampled[sample_offset : sample_offset + available],
            1 - phase,
            out=result[:available],
        )
        result[:available] += phase * resampled[sample_offset + 1 : sample_offset + available + 1]
    if sample_count - available > 1:
        raise ValueError("Resampling window does not cover the full target waveform")
    if available < sample_count:
        result[available:] = resampled[-1]
    return result


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    frames: int
    channels: int
    subtype: str
    dc_mean: float

    @property
    def duration_seconds(self) -> float:
        return self.frames / self.sample_rate


class AudioProcessor:
    def __init__(self, path: Path, config: MR750Config):
        self.path = path
        self.config = config
        self.info = self._inspect_audio()
        self.output_samples = round(self.info.duration_seconds * config.output_sample_rate)
        if self.output_samples < 2:
            raise ValueError("Audio is too short to create a gradient event")
        self.segment_samples = config.segment_seconds * config.output_sample_rate
        self.filter = butter(
            6, config.lowpass_hz, btype="lowpass", fs=self.info.sample_rate, output="sos"
        )
        divisor = math.gcd(self.info.sample_rate, config.output_sample_rate)
        self.resample_up = config.output_sample_rate // divisor
        self.resample_down = self.info.sample_rate // divisor

    def _inspect_audio(self) -> AudioInfo:
        try:
            with sf.SoundFile(self.path) as audio_file:
                if audio_file.format != "WAV" or not audio_file.subtype.startswith("PCM_"):
                    raise ValueError("Input must be a PCM WAV file")
                if audio_file.channels not in (1, 2):
                    raise ValueError("Only mono or stereo WAV files are supported")
                if audio_file.frames < 2:
                    raise ValueError("WAV file does not contain enough audio samples")
                if audio_file.samplerate <= 2 * self.config.lowpass_hz:
                    raise ValueError("WAV sample rate must exceed twice the low-pass cutoff")
                sample_sum = 0.0
                for frames in audio_file.blocks(
                    blocksize=1_000_000, dtype="float64", always_2d=True
                ):
                    if not np.all(np.isfinite(frames)):
                        raise ValueError("WAV contains non-finite samples")
                    sample_sum += float(np.sum(np.mean(frames, axis=1)))
                return AudioInfo(
                    sample_rate=audio_file.samplerate,
                    frames=audio_file.frames,
                    channels=audio_file.channels,
                    subtype=audio_file.subtype,
                    dc_mean=sample_sum / audio_file.frames,
                )
        except (OSError, RuntimeError) as error:
            raise ValueError(f"Unable to read WAV file {self.path}: {error}") from error

    def segment_bounds(self) -> list[tuple[int, int]]:
        return [
            (start, min(start + self.segment_samples, self.output_samples))
            for start in range(0, self.output_samples, self.segment_samples)
        ]

    def read_segment(self, start_sample: int, end_sample: int) -> np.ndarray:
        output_rate = self.config.output_sample_rate
        input_rate = self.info.sample_rate
        start_time = start_sample / output_rate
        end_time = end_sample / output_rate
        window_start = max(0, math.floor((start_time - WINDOW_MARGIN_SECONDS) * input_rate))
        # Align the window to keep resampling phase consistent across segments
        window_start -= window_start % self.resample_down
        window_end = min(
            self.info.frames, math.ceil((end_time + WINDOW_MARGIN_SECONDS) * input_rate)
        )

        with sf.SoundFile(self.path) as audio_file:
            audio_file.seek(window_start)
            frames = audio_file.read(window_end - window_start, dtype="float64", always_2d=True)

        mono = np.mean(frames, axis=1) - self.info.dc_mean
        filtered = sosfiltfilt(self.filter, mono, padlen=min(30, len(mono) - 1))
        resampled = resample_poly(filtered, self.resample_up, self.resample_down)
        result = align_resampled_samples(
            resampled, start_sample, end_sample, window_start, input_rate, output_rate
        )

        fade_samples = min(round(self.config.fade_ms * output_rate / 1000), len(result) // 2)
        if fade_samples > 0:
            fade = np.sin(np.linspace(0, np.pi / 2, fade_samples)) ** 2
            result[:fade_samples] *= fade
            result[-fade_samples:] *= fade[::-1]
        return result


def measure_waveform(waveform: np.ndarray, raster_seconds: float) -> tuple[float, float]:
    peak = float(np.max(np.abs(waveform)))
    slew_per_second = max(
        float(np.max(np.abs(np.diff(waveform)))) / raster_seconds,
        2 * abs(float(waveform[0])) / raster_seconds,
        2 * abs(float(waveform[-1])) / raster_seconds,
    )
    return peak, slew_per_second

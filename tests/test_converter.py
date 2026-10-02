import json
from pathlib import Path

import numpy as np
import pypulseq as pp
import pytest
import soundfile as sf

from mri_player.audio import AudioProcessor, align_resampled_samples
from mri_player.config import load_config
from mri_player.converter import convert
from mri_player.sequence import SequenceMetrics, write_sequence


def config_file(tmp_path: Path, **overrides: str | float) -> Path:
    values = {
        "scanner_model": '"GE Discovery MR750 3.0T"',
        "profile_name": '"test"',
        "gradient_raster_us": 100,
        "lowpass_hz": 2000,
        "max_gradient_mtm": 1.0,
        "max_slew_tms": 10.0,
        "target_peak_mtm": 0.1,
        "segment_seconds": 1,
        "fade_ms": 20,
    }
    values.update(overrides)
    path = tmp_path / "config.toml"
    path.write_text("\n".join(f"{key} = {value}" for key, value in values.items()) + "\n")
    return path


def tone(sample_rate: int, seconds: float, frequency: float = 440) -> np.ndarray:
    time = np.arange(round(sample_rate * seconds)) / sample_rate
    return 0.4 * np.sin(2 * np.pi * frequency * time)


def test_split_stereo_and_roundtrip(tmp_path: Path) -> None:
    sample_rate = 8000
    samples = np.column_stack((tone(sample_rate, 2.2), tone(sample_rate, 2.2, 660)))
    input_path = tmp_path / "music.wav"
    sf.write(input_path, samples, sample_rate, subtype="PCM_16")
    config = load_config(config_file(tmp_path))

    manifest = convert(input_path, config, tmp_path / "out")

    assert manifest["scanner_model"] == "GE Discovery MR750 3.0T"
    assert manifest["channel_mapping"] == "stereo_average_to_x"
    assert len(manifest["segments"]) == 3
    assert [segment["source_start_seconds"] for segment in manifest["segments"]] == [0, 1, 2]
    assert manifest["segments"][-1]["source_end_seconds"] == pytest.approx(2.2)
    assert sum(
        segment["sequence_duration_seconds"] for segment in manifest["segments"]
    ) == pytest.approx(2.2)
    assert json.loads((tmp_path / "out" / "manifest.json").read_text()) == manifest

    for segment in manifest["segments"]:
        assert segment["max_gradient_mtm"] <= config.max_gradient_mtm
        assert segment["max_slew_tms"] <= config.max_slew_tms
        sequence = pp.Sequence()
        sequence.read(str(tmp_path / "out" / segment["file"]))
        assert sequence.check_timing()[0]
        assert sequence.duration()[0] == pytest.approx(segment["sequence_duration_seconds"])
        for block_index in range(1, segment["blocks"] + 1):
            block = sequence.get_block(block_index)
            assert block.gx is not None
            assert block.gy is None
            assert block.gz is None
            assert block.rf is None
            assert block.adc is None
        assert sequence.get_block(1).gx.first == pytest.approx(0)
        assert sequence.get_block(segment["blocks"]).gx.last == pytest.approx(0)


def test_stereo_average_matches_mono(tmp_path: Path) -> None:
    sample_rate = 8000
    samples = tone(sample_rate, 0.2)
    mono_path = tmp_path / "mono.wav"
    stereo_path = tmp_path / "stereo.wav"
    sf.write(mono_path, samples, sample_rate, subtype="PCM_16")
    sf.write(stereo_path, np.column_stack((samples, samples)), sample_rate, subtype="PCM_16")
    config = load_config(config_file(tmp_path))

    mono = AudioProcessor(mono_path, config).read_segment(0, 2000)
    stereo = AudioProcessor(stereo_path, config).read_segment(0, 2000)

    np.testing.assert_allclose(mono, stereo, atol=1e-12)


def test_tone_frequency_survives_audio_processing(tmp_path: Path) -> None:
    input_path = tmp_path / "tone.wav"
    sf.write(input_path, tone(8000, 0.4), 8000, subtype="PCM_16")
    config = load_config(config_file(tmp_path))

    processed = AudioProcessor(input_path, config).read_segment(0, 4000)
    central_samples = processed[200:-200]
    frequencies = np.fft.rfftfreq(len(central_samples), d=config.raster_seconds)
    dominant_frequency = frequencies[np.argmax(np.abs(np.fft.rfft(central_samples)))]

    assert dominant_frequency == pytest.approx(440, abs=5)


@pytest.mark.parametrize(
    ("input_rate", "output_rate", "start_sample", "window_start", "resampled_count", "count"),
    [
        (44100, 100000, 10000, 3000, 10000, 1000),
        (44117, 100000, 10000, 3000, 10000, 1000),
        (8000, 10000, 0, 0, 100, 100),
    ],
)
def test_sample_alignment_matches_time_interpolation(
    input_rate: int,
    output_rate: int,
    start_sample: int,
    window_start: int,
    resampled_count: int,
    count: int,
) -> None:
    values = np.sin(np.arange(resampled_count) * 0.03)
    wanted_times = (
        np.arange(start_sample, start_sample + count) + 0.5
    ) / output_rate - window_start / input_rate
    reference = np.interp(wanted_times, np.arange(resampled_count) / output_rate, values)

    result = align_resampled_samples(
        values, start_sample, start_sample + count, window_start, input_rate, output_rate
    )

    np.testing.assert_allclose(result, reference, atol=1e-12)


def test_adjacent_segments_match_whole_audio_away_from_fades(tmp_path: Path) -> None:
    input_rate = 44117
    input_path = tmp_path / "awkward-rate.wav"
    sf.write(input_path, tone(input_rate, 1.3), input_rate, subtype="PCM_16")
    config = load_config(config_file(tmp_path, fade_ms=0))
    processor = AudioProcessor(input_path, config)
    split = np.concatenate(
        [processor.read_segment(start, end) for start, end in processor.segment_bounds()]
    )
    whole = processor.read_segment(0, processor.output_samples)

    np.testing.assert_allclose(split, whole, atol=1e-7)


def test_silence_and_invalid_audio(tmp_path: Path) -> None:
    input_path = tmp_path / "silent.wav"
    sf.write(input_path, np.zeros(800), 8000, subtype="PCM_16")
    config = load_config(config_file(tmp_path))
    manifest = convert(input_path, config, tmp_path / "silent-out")
    assert manifest["global_scale_mtm_per_audio_unit"] == 0
    assert manifest["segments"][0]["max_gradient_mtm"] == 0

    invalid_path = tmp_path / "invalid.wav"
    invalid_path.write_text("not a wave file")
    with pytest.raises(ValueError, match="Unable to read WAV"):
        convert(invalid_path, config, tmp_path / "invalid-out")
    assert not (tmp_path / "invalid-out").exists()

    three_channel_path = tmp_path / "three.wav"
    sf.write(three_channel_path, np.zeros((800, 3)), 8000, subtype="PCM_16")
    with pytest.raises(ValueError, match="Only mono or stereo"):
        convert(three_channel_path, config, tmp_path / "three-out")


def test_opposite_stereo_channels_do_not_amplify_quantization_noise(tmp_path: Path) -> None:
    samples = tone(8000, 0.2)
    input_path = tmp_path / "opposite.wav"
    sf.write(input_path, np.column_stack((samples, -samples)), 8000, subtype="PCM_16")
    config = load_config(config_file(tmp_path))

    manifest = convert(input_path, config, tmp_path / "opposite-out")

    assert manifest["global_scale_mtm_per_audio_unit"] == 0
    assert manifest["segments"][0]["max_gradient_mtm"] == 0


def test_mr750_config_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Only GE Discovery MR750"):
        load_config(config_file(tmp_path, scanner_model='"Other Scanner"'))
    with pytest.raises(ValueError, match="published MR750 limit of 50 mT/m"):
        load_config(config_file(tmp_path, max_gradient_mtm=51))
    with pytest.raises(ValueError, match="published MR750 limit of 200 T/m/s"):
        load_config(config_file(tmp_path, max_slew_tms=201))


def test_failed_conversion_leaves_no_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_path = tmp_path / "music.wav"
    sf.write(input_path, tone(8000, 1.1), 8000, subtype="PCM_16")
    config = load_config(config_file(tmp_path))

    def fail_on_second_segment(waveform: np.ndarray, config: object, path: Path) -> SequenceMetrics:
        if path.name == "segment-0002.seq":
            raise ValueError("Simulated failure on segment 2")
        return write_sequence(waveform, config, path)

    monkeypatch.setattr("mri_player.converter.write_sequence", fail_on_second_segment)
    with pytest.raises(ValueError, match="Simulated failure on segment 2"):
        convert(input_path, config, tmp_path / "out")
    assert not (tmp_path / "out").exists()

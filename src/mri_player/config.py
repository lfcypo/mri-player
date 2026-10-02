import math
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path

SCANNER_MODEL = "GE Discovery MR750 3.0T"
PUBLIC_MAX_GRADIENT_MTM = 50.0
PUBLIC_MAX_SLEW_TMS = 200.0


@dataclass(frozen=True)
class MR750Config:
    scanner_model: str
    profile_name: str
    gradient_raster_us: int
    lowpass_hz: float
    max_gradient_mtm: float
    max_slew_tms: float
    target_peak_mtm: float
    segment_seconds: int
    fade_ms: float

    @property
    def output_sample_rate(self) -> int:
        return 1_000_000 // self.gradient_raster_us

    @property
    def raster_seconds(self) -> float:
        return self.gradient_raster_us / 1_000_000

    def as_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def load_config(path: Path) -> MR750Config:
    try:
        with path.open("rb") as config_file:
            data = tomllib.load(config_file)
        config = MR750Config(**data)
    except (OSError, tomllib.TOMLDecodeError, TypeError) as error:
        raise ValueError(f"Unable to read configuration {path}: {error}") from error

    if config.scanner_model != SCANNER_MODEL:
        raise ValueError(f"Only {SCANNER_MODEL} is supported")
    if not config.profile_name.strip():
        raise ValueError("profile_name must not be empty")
    if (
        isinstance(config.gradient_raster_us, bool)
        or not isinstance(config.gradient_raster_us, int)
        or config.gradient_raster_us <= 0
        or 1_000_000 % config.gradient_raster_us != 0
    ):
        raise ValueError("gradient_raster_us must be a positive integer divisor of 1,000,000")
    for name in ("lowpass_hz", "max_gradient_mtm", "max_slew_tms", "target_peak_mtm"):
        value = getattr(config, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be a finite positive number")
    if config.max_gradient_mtm > PUBLIC_MAX_GRADIENT_MTM:
        raise ValueError("max_gradient_mtm exceeds the published MR750 limit of 50 mT/m")
    if config.max_slew_tms > PUBLIC_MAX_SLEW_TMS:
        raise ValueError("max_slew_tms exceeds the published MR750 limit of 200 T/m/s")
    if config.target_peak_mtm > config.max_gradient_mtm:
        raise ValueError("target_peak_mtm must not exceed max_gradient_mtm")
    if config.lowpass_hz >= config.output_sample_rate / 2:
        raise ValueError("lowpass_hz must be below half the gradient sample rate")
    if (
        isinstance(config.segment_seconds, bool)
        or not isinstance(config.segment_seconds, int)
        or config.segment_seconds <= 0
    ):
        raise ValueError("segment_seconds must be a positive integer")
    if (
        isinstance(config.fade_ms, bool)
        or not isinstance(config.fade_ms, (int, float))
        or not math.isfinite(config.fade_ms)
        or config.fade_ms < 0
        or config.fade_ms * 2 >= config.segment_seconds * 1000
    ):
        raise ValueError("fade_ms must be nonnegative and both fades must fit within a segment")
    return config

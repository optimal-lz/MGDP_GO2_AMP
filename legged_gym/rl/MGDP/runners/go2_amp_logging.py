"""Small, strict TensorBoard scalar logger for the formal Go2 AMP task."""

import math


class IterationMetrics:
    """Write one finite scalar set per monotonically increasing iteration."""

    def __init__(self, writer, mode, start_step=-1):
        if mode not in ("amp", "no_amp"):
            raise ValueError("mode must be 'amp' or 'no_amp'")
        self.writer = writer
        self.mode = mode
        self.last_step = int(start_step)

    @staticmethod
    def _scalar(tag, value):
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"metric {tag!r} must be finite")
        return value

    def write(self, iteration, core, detail=None):
        iteration = int(iteration)
        if iteration <= self.last_step:
            raise ValueError("metric iterations must be strictly monotonic")

        values = dict(core)
        if detail is not None and iteration % 10 == 0:
            values.update(detail)
        scalars = {
            tag: self._scalar(tag, value) for tag, value in values.items()
        }

        if self.mode == "no_amp":
            contribution = scalars.get("reward/amp_contribution", 0.0)
            if contribution != 0.0:
                raise ValueError("noAMP reward contribution must remain zero")
            if any("discriminator" in tag.lower() for tag in scalars):
                raise ValueError("noAMP cannot log discriminator training metrics")

        if self.writer is not None:
            for tag, value in scalars.items():
                self.writer.add_scalar(tag, value, iteration)
        self.last_step = iteration


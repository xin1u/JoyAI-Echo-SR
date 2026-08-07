import torch


class TimestepSampler:
    """Base class for timestep samplers.
    Timestep samplers are used to sample timesteps for diffusion models.
    They should implement both sample() and sample_for() methods.
    """

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:
        """Sample timesteps for a batch.
        Args:
            batch_size: Number of timesteps to sample
            seq_length: (optional) Length of the sequence being processed
            device: Device to place the samples on
        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.
        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)
        Returns:
            Tensor of shape (batch_size,) containing timesteps
        """
        raise NotImplementedError


class UniformTimestepSampler(TimestepSampler):
    """Samples timesteps uniformly between min_value and max_value (default 0 and 1)."""

    def __init__(self, min_value: float = 0.0, max_value: float = 1.0):
        self.min_value = min_value
        self.max_value = max_value

    def sample(self, batch_size: int, seq_length: int | None = None, device: torch.device = None) -> torch.Tensor:  # noqa: ARG002
        return torch.rand(batch_size, device=device) * (self.max_value - self.min_value) + self.min_value

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        return self.sample(batch.shape[0], device=batch.device)


class ShiftedLogitNormalTimestepSampler(TimestepSampler):
    """
    Samples timesteps from a stretched shifted logit-normal distribution,
    where the shift is determined by the sequence length.
    The stretching normalizes samples between percentile bounds to ensure
    the distribution covers [0, 1] more evenly. A uniform fallback prevents
    collapse at high token counts.
    """

    def __init__(self, std: float = 1.0, eps: float = 1e-3, uniform_prob: float = 0.1, shift: float | None = None):
        self.std = std
        self.eps = eps
        self.uniform_prob = uniform_prob
        self.fixed_shift = shift
        self.normal_999_percentile = 3.0902 * std
        self.normal_005_percentile = -2.5758 * std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        mu = self.fixed_shift if self.fixed_shift is not None else self._get_shift_for_sequence_length(seq_length)

        normal_samples = torch.randn((batch_size,), device=device) * self.std + mu
        logitnormal_samples = torch.sigmoid(normal_samples)

        percentile_999 = torch.sigmoid(torch.tensor(mu + self.normal_999_percentile, device=device))
        percentile_005 = torch.sigmoid(torch.tensor(mu + self.normal_005_percentile, device=device))

        zero_terminal_raw = (logitnormal_samples - percentile_005) / (percentile_999 - percentile_005)

        stretched_logit = torch.where(
            zero_terminal_raw >= self.eps,
            zero_terminal_raw,
            2 * self.eps - zero_terminal_raw,
        )
        stretched_logit = torch.clamp(stretched_logit, 0, 1)

        uniform = (1 - self.eps) * torch.rand((batch_size,), device=device) + self.eps
        prob = torch.rand((batch_size,), device=device)

        return torch.where(prob > self.uniform_prob, stretched_logit, uniform)

    def sample_for(self, batch: torch.Tensor) -> torch.Tensor:
        """Sample timesteps for a specific batch tensor.
        Args:
            batch: Input tensor of shape (batch_size, seq_length, ...)
        Returns:
            Tensor of shape (batch_size,) containing timesteps sampled from a shifted
            logit-normal distribution, where the shift is determined by the sequence length
            of the input batch
        Raises:
            ValueError: If the input batch does not have 3 dimensions
        """
        if batch.ndim != 3:
            raise ValueError(f"Batch should have 3 dimensions, got {batch.ndim}")

        batch_size, seq_length, _ = batch.shape
        return self.sample(batch_size, seq_length, device=batch.device)

    @staticmethod
    def _get_shift_for_sequence_length(
        seq_length: int,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> float:
        m = (max_shift - min_shift) / (max_tokens - min_tokens)
        b = min_shift - m * min_tokens
        shift = m * seq_length + b
        # Cap at 13 — the 1K SR value (1920×1152, 34560 tokens). Higher shifts
        # cause float32 sigmoid saturation, and stretching produces identical
        # distributions anyway, so there's no benefit to larger values.
        return min(shift, 13.0)


SAMPLERS = {
    "uniform": UniformTimestepSampler,
    "shifted_logit_normal": ShiftedLogitNormalTimestepSampler,
}


def example() -> None:
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sampler = ShiftedLogitNormalTimestepSampler()
    for seq_length in [1024, 2048, 4096, 8192]:
        samples = sampler.sample(batch_size=1_000_000, seq_length=seq_length)

        # plot the histogram of the samples
        plt.hist(samples.numpy(), bins=100, density=True)
        plt.title(f"Timestep Samples for Sequence Length {seq_length}")
        plt.xlabel("Timestep")
        plt.ylabel("Density")
        plt.show()


if __name__ == "__main__":
    example()

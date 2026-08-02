from __future__ import annotations

import torch


class RoPE(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        assert head_dim % 2 == 0
        # inv_freq_i = 1 / base ^ { 2i / head_dim }, [head_dim/2]
        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        # x: [B, H, L, head_dim]/ [L]
        batch_size, _, sequence_length, _ = x.shape

        if position_ids.ndim == 1:
            # [L]
            assert position_ids.shape == (sequence_length,), "1D position_ids must have shape [L]"
            # [1, L]
            position_ids = position_ids.unsqueeze(0)
        elif position_ids.ndim == 2:
            assert position_ids.shape in (
                (1, sequence_length),
                (batch_size, sequence_length),
            ), "2D position_ids must have shape [1, L] or [B, L]"
        else:
            raise ValueError("position_ids must have shape [L] or [B, L]")
        # [B, L]
        position_ids = position_ids.to(device=x.device, dtype=self.inv_freq.dtype)

        # position_ids:     [B or 1, L] -> [B or 1, L, 1]
        # inv_freq:         [head_dim/2] -> [1, 1, head_dim/2]
        # broadcast angles: [B or 1, L, head_dim/2]
        angles = position_ids.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        # [B or 1, 1, L, head_dim/2]
        cos = angles.cos().unsqueeze(1)
        sin = angles.sin().unsqueeze(1)

        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        # [B, H, L, head_dim/2]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # [B, H, L, head_dim/2, 2]
        rotated = torch.stack((rotated_even, rotated_odd), dim=-1)
        # [B, H, L, head_dim]
        return rotated.flatten(-2)
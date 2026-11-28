from torch import nn


class BaseModel(nn.Module):
    """Inference-only base for the upstream Apollo architecture."""

    def __init__(self, sample_rate):
        super().__init__()
        self.sample_rate = sample_rate

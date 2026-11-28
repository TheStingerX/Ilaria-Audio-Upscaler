# Adapted from JusperLee/Apollo inference.py (CC BY-SA 4.0).
import torch
import torch.nn.functional as F

def validate_chunk_batch_size(chunk_batch_size):
    """Require a fixed positive number of chunks per model forward pass."""
    if (
        isinstance(chunk_batch_size, bool)
        or not isinstance(chunk_batch_size, int)
        or chunk_batch_size < 1
    ):
        raise ValueError("Chunk batch size must be a positive integer.")
    return chunk_batch_size


def chunk_starts(total_samples, chunk_samples, overlap_samples):
    """Return starts that cover the signal without a redundant final chunk."""
    hop_samples = chunk_samples - overlap_samples
    starts = [0]
    while starts[-1] + chunk_samples < total_samples:
        starts.append(starts[-1] + hop_samples)
    return starts


def crossfade_weights(length, overlap_samples, fade_in, fade_out, dtype):
    """Create linear edge weights for normalized overlap-add."""
    weights = torch.ones(length, dtype=dtype)
    fade_samples = min(overlap_samples, length)
    if fade_samples:
        ramp = torch.linspace(0.0, 1.0, fade_samples, dtype=dtype)
        if fade_in:
            weights[:fade_samples] = ramp
        if fade_out:
            weights[-fade_samples:] = torch.flip(ramp, dims=(0,))
    return weights.view(1, 1, -1)


def run_model(
    model,
    audio,
    device,
    chunk_samples=None,
    overlap_samples=0,
    chunk_batch_size=1,
    chunk_pad_samples=0,
):
    """Run full-file or batched normalized overlap-add chunked inference.

    Each chunk is inferred with `chunk_pad_samples` of surrounding audio per
    side, discarded from the output, because model output is wrong near the
    edges of its input.
    """
    chunk_batch_size = validate_chunk_batch_size(chunk_batch_size)

    total_samples = audio.shape[-1]
    if chunk_samples is None or total_samples <= chunk_samples:
        return model(audio.to(device)).detach().to("cpu")

    output_sum = torch.zeros_like(audio, device="cpu")
    weight_sum = torch.zeros((1, 1, total_samples), dtype=audio.dtype)
    starts = chunk_starts(total_samples, chunk_samples, overlap_samples)
    padded_chunk_samples = chunk_samples + 2 * chunk_pad_samples

    for batch_start in range(0, len(starts), chunk_batch_size):
        batch_starts = starts[batch_start : batch_start + chunk_batch_size]
        batch_chunks = []
        valid_lengths = []
        batch_padded_starts = []
        for start in batch_starts:
            end = min(start + chunk_samples, total_samples)
            valid_samples = end - start
            # Real audio pads each side. The file's own edges take none:
            # the first padded chunk begins at the file start and the last
            # ends at the file end, extending further left instead —
            # padding with silence measurably degrades the output near it.
            padded_start = max(0, start - chunk_pad_samples)
            if chunk_pad_samples and end == total_samples:
                padded_start = max(0, total_samples - padded_chunk_samples)
            chunk = audio[..., padded_start : padded_start + padded_chunk_samples]
            if chunk.shape[-1] < padded_chunk_samples:
                chunk = F.pad(chunk, (0, padded_chunk_samples - chunk.shape[-1]))
            batch_chunks.append(chunk)
            valid_lengths.append(valid_samples)
            batch_padded_starts.append(padded_start)

        chunk_batch = torch.cat(batch_chunks, dim=0)
        batch_output = model(chunk_batch.to(device)).detach().to("cpu")
        expected_shape = (len(batch_starts), audio.shape[1])
        if batch_output.ndim != 3 or batch_output.shape[:2] != expected_shape:
            raise RuntimeError(
                "Chunk output must preserve the input batch and channel dimensions."
            )
        if batch_output.shape[-1] < padded_chunk_samples:
            raise RuntimeError(
                "Chunk output is shorter than the corresponding input segment."
            )

        for offset, (start, valid_samples, padded_start) in enumerate(
            zip(batch_starts, valid_lengths, batch_padded_starts)
        ):
            end = start + valid_samples
            valid_start = start - padded_start
            chunk_output = batch_output[
                offset : offset + 1, ..., valid_start : valid_start + valid_samples
            ]
            index = batch_start + offset
            weights = crossfade_weights(
                valid_samples,
                overlap_samples,
                fade_in=index > 0,
                fade_out=index < len(starts) - 1,
                dtype=audio.dtype,
            )
            output_sum[..., start:end] += chunk_output * weights
            weight_sum[..., start:end] += weights

    if torch.any(weight_sum <= 0):
        raise RuntimeError("Chunk overlap-add left uncovered output samples.")
    return output_sum / weight_sum

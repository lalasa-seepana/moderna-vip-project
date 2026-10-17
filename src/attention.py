import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'src'

"""Untrained, single-head self-attention; no biological interpretation is implied."""
from .common import OUTPUT
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VOCAB = 'ACGU'
EMBED_DIM = 8


def encode_sequence(sequence, embed_dim=EMBED_DIM):
    # The PRD's acceptance example uses four bases; the application enforces 5–50.
    if not isinstance(sequence, str) or not 1 <= len(sequence) <= 50:
        raise ValueError('Sequence must contain 1–50 bases; application input requires 5–50')
    invalid = set(sequence) - set(VOCAB)
    if invalid:
        raise ValueError(f'Invalid nucleotide(s): {sorted(invalid)}; expected A, C, G, U')
    table = np.random.default_rng(42).normal(size=(4, embed_dim))
    return table[[VOCAB.index(base) for base in sequence]]


def projections(X, d_k=EMBED_DIM):
    if X.ndim != 2 or not X.size or not np.isfinite(X).all() or d_k < 1:
        raise ValueError('Expected a finite, nonempty matrix and positive d_k')
    rng = np.random.default_rng(43)
    return tuple(X @ rng.normal(0, .1, (X.shape[1], d_k)) for _ in range(3))


def self_attention(X, d_k=EMBED_DIM):
    q, k, v = projections(X, d_k)
    scores = q @ k.T / np.sqrt(d_k)
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights, weights @ v


def torch_self_attention(X, d_k=EMBED_DIM):
    import torch
    q, k, v = [torch.from_numpy(a) for a in projections(X, d_k)]
    weights = torch.softmax(q @ k.T / np.sqrt(d_k), dim=-1)
    return weights.numpy(), (weights @ v).numpy()


def label_heatmap(ax, weights, sequence):
    image = ax.imshow(weights, cmap='viridis')
    labels = [f'{i+1}:{base}' for i, base in enumerate(sequence)]
    ax.set_xticks(range(len(sequence)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(sequence)), labels, fontsize=7)
    ax.set(xlabel='Attending to position', ylabel='Attending from position', title='Untrained mRNA self-attention')
    return image


def run(sequence='ACGUACGUACGGCUA'):
    if not 5 <= len(sequence) <= 50:
        raise ValueError('Application sequence length must be 5–50')
    X = encode_sequence(sequence)
    weights, values = self_attention(X)
    tw, tv = torch_self_attention(X)
    if not np.allclose(weights.sum(axis=1), 1.0):
        raise RuntimeError('Attention rows must sum to one')
    if not (np.allclose(weights, tw) and np.allclose(values, tv)):
        raise RuntimeError('NumPy and PyTorch attention disagree')
    (OUTPUT / 'figures').mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT / 'attention.npz', weights=weights, output=values, sequence=sequence)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = label_heatmap(ax, weights, sequence)
    fig.colorbar(image, ax=ax, label='Attention weight')
    fig.tight_layout(); fig.savefig(OUTPUT / 'figures/attention_heatmap.png', dpi=160); plt.close(fig)
    return weights


if __name__ == '__main__':
    run()

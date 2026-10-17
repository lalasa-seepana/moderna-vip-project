import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'src'

from .common import OUTPUT, read_json
from .attention import label_heatmap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def build_dashboard(output=OUTPUT):
    metrics = read_json(output / 'figures/model_metrics.json')
    selected = metrics['selected']
    roc = metrics['models'][selected]['roc'] | {'auc': metrics['models'][selected]['auc'], 'model': selected}
    attention = np.load(output / 'attention.npz', allow_pickle=False)
    timing = read_json(output / 'latencies.json')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].plot(roc['fpr'], roc['tpr'], color='#137f83', lw=2, label=f"AUC = {roc['auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], '--', color='gray')
    axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title=f"Risk classifier ROC\n{roc['model'].replace('_', ' ')}")
    axes[0].legend(loc='lower right')
    label_heatmap(axes[1], attention['weights'], str(attention['sequence']))
    labels = ['RAG query', 'LLM report']
    values = [timing['rag_seconds'], timing['report_seconds']]
    bars = axes[2].bar(labels, values, color=['#137f83', '#384d76'])
    axes[2].bar_label(bars, labels=[f'{v:.4f} s' for v in values], padding=4)
    axes[2].set(title='Measured stage latency', ylabel='Seconds')
    axes[2].set_ylim(0, max(values) * 1.3)
    fig.suptitle('mRNA Clinical Intelligence Educational dashboard', fontsize=15)
    fig.tight_layout(); fig.savefig(output / 'figures/dashboard.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    build_dashboard()

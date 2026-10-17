"""Run the five PRD modules: python -m src.pipeline."""
from .risk_model import train
from .attention import run as run_attention
from .rag_index import build_index, evaluate
from .report_generator import run_reports
from .dashboard import build_dashboard
from .portfolio import build_portfolio


def main():
    print('1/5 Training risk classifiers...', flush=True)
    selected, models = train()
    print('Selected:', selected, flush=True)
    print('2/5 Computing self-attention...', flush=True)
    run_attention()
    print('3/5 Building or loading the local semantic index...', flush=True)
    index = build_index()
    evaluate(index)
    print('4/5 Generating five LLM reports...', flush=True)
    run_reports(index)
    print('5/5 Building dashboard and project report...', flush=True)
    build_dashboard()
    build_portfolio()
    print('Pipeline completed.', flush=True)


if __name__ == '__main__':
    main()

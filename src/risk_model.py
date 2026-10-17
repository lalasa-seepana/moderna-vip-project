import sys
from pathlib import Path
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'src'

from .common import DATA, OUTPUT, save_json
from .data_pipeline import load_records, clean_records, FEATURES, NUMERIC_COLS, CATEGORICAL_COL, TARGET
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def preprocessor():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUMERIC_COLS),
        ('history', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='error', sparse_output=False))]), [CATEGORICAL_COL]),
        ('conditions', 'passthrough', ['Pre_Existing_Conditions']),
    ])


def train(path=DATA / 'patient_records.csv', output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    figures = output / 'figures'
    figures.mkdir(exist_ok=True)
    df = load_records(path)
    train_df, test_df = train_test_split(df, test_size=.2, stratify=df[TARGET], random_state=42)
    clean = clean_records(df, train_df)
    clean.to_csv(DATA / 'patient_records_clean.csv', index=False)
    metrics, models = {}, {}
    for name, estimator in [('logistic_regression', LogisticRegression(max_iter=2000, random_state=42)),
                            ('random_forest', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))]:
        model = Pipeline([('preprocess', preprocessor()), ('classifier', estimator)])
        model.fit(train_df[FEATURES], train_df[TARGET])
        pred = model.predict(test_df[FEATURES])
        scores = model.predict_proba(test_df[FEATURES])[:, 1]
        fpr, tpr, _ = roc_curve(test_df[TARGET], scores)
        metrics[name] = {'f1': float(f1_score(test_df[TARGET], pred)), 'auc': float(roc_auc_score(test_df[TARGET], scores)),
            'classification_report': classification_report(test_df[TARGET], pred, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(test_df[TARGET], pred, labels=[0, 1]).tolist(),
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}}
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(metrics[name]['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
        ax.set(xlabel='Predicted risk', ylabel='Observed class', title=name.replace('_', ' ').title())
        fig.tight_layout(); fig.savefig(figures / f'{name}_confusion_matrix.png', dpi=160); plt.close(fig)
        models[name] = model
    selected = max(metrics, key=lambda name: metrics[name]['f1'])
    model = models[selected]
    joblib.dump(model, output / 'risk_model.joblib')
    save_json(figures / 'model_metrics.json', {'selected': selected, 'models': metrics})
    return selected, models


def predict_patient(patient, model_path=OUTPUT / 'risk_model.joblib'):
    import pandas as pd
    model = joblib.load(model_path)
    frame = pd.DataFrame([patient])[FEATURES]
    probability = float(model.predict_proba(frame)[0, 1])
    label = int(model.predict(frame)[0])
    return {'risk_label': 'High' if label else 'Low', 'high_risk_probability': probability, 'patient': {k: patient[k] for k in FEATURES}}


if __name__ == '__main__':
    print('Selected:', train()[0])

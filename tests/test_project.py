import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from src.common import DATA, OUTPUT, DISCLAIMER, read_json
from src.data_pipeline import load_records, load_and_clean, DataValidationError, FEATURES, NUMERIC_COLS
from src.attention import encode_sequence, self_attention, torch_self_attention, run
from src.schemas import ClinicalSafetyReport
from src.report_generator import generate_report, apply_safety_guards, explicit_maximum_mg, ReportGenerationError, build_prompt, risk_statement


def risk():
    return {'risk_label': 'Low', 'high_risk_probability': .2, 'patient': {'Age': 58, 'Biomarker_Level': 3.0, 'Vaccine_Dose_mcg': 120., 'Pre_Existing_Conditions': 2, 'Prior_Reaction_History': 'None'}}


def valid(chunks=None):
    return {'Eligible_For_Vaccine': False, 'Recommended_Dose_mg': .12, 'Reasoning_Steps': [risk_statement(risk())], 'FDA_Safety_Warnings_Cited': chunks if chunks is not None else ['Maximum safe dose is 100 micrograms.'], 'Disclaimer': DISCLAIMER}


@pytest.mark.parametrize('column', ['Patient_ID'] + FEATURES + ['Reaction_Score'])
def test_missing_columns(tmp_path, column):
    df = load_records().drop(columns=[column]); path = tmp_path / 'bad.csv'; df.to_csv(path, index=False)
    with pytest.raises(DataValidationError, match=column): load_records(path)


def test_cleaning_preserves_none():
    raw = load_records()
    assert (raw.Prior_Reaction_History == 'None').sum() == 291
    assert raw.Prior_Reaction_History.isna().sum() == 17
    assert raw.Biomarker_Level.isna().sum() == 20
    clean = load_and_clean()
    assert not clean.isna().any().any()
    assert clean.loc[raw.Biomarker_Level.isna(), 'Biomarker_Level'].eq(raw.Biomarker_Level.median()).all()


@pytest.mark.parametrize('column,value', [('Age', -1), ('Age', 1.2), ('Biomarker_Level', float('inf')), ('Reaction_Score', 4), ('Prior_Reaction_History', 'Unknown')])
def test_bad_data(tmp_path, column, value):
    df = load_records(); df[column] = df[column].astype(object); df.loc[0, column] = value
    path = tmp_path / 'bad.csv'; df.to_csv(path, index=False)
    with pytest.raises(DataValidationError): load_records(path)


def test_split_scaling_and_serialization():
    import joblib
    from sklearn.model_selection import train_test_split
    df = load_records()
    train, test = train_test_split(df, test_size=.2, stratify=df.Reaction_Score, random_state=42)
    train2, test2 = train_test_split(df, test_size=.2, stratify=df.Reaction_Score, random_state=42)
    assert train.Patient_ID.tolist() == train2.Patient_ID.tolist()
    assert test.Patient_ID.tolist() == test2.Patient_ID.tolist()
    assert len(train) == 400 and len(test) == 100
    assert not set(train.Patient_ID) & set(test.Patient_ID)
    model = joblib.load(OUTPUT / 'risk_model.joblib')
    X = model.named_steps['preprocess'].transform(train[FEATURES])[:, :3]
    np.testing.assert_allclose(X.mean(0), 0, atol=1e-12)
    np.testing.assert_allclose(X.std(0), 1, atol=1e-12)
    imputer = model.named_steps['preprocess'].named_transformers_['numeric'].named_steps['impute']
    np.testing.assert_allclose(imputer.statistics_, train[NUMERIC_COLS].median())
    from sklearn.metrics import roc_curve, f1_score
    selection = read_json(OUTPUT / 'figures/model_metrics.json')
    predicted = model.predict(test[FEATURES])
    scores = model.predict_proba(test[FEATURES])[:, 1]
    fpr, tpr, _ = roc_curve(test.Reaction_Score, scores)
    selected = selection['models'][selection['selected']]
    np.testing.assert_allclose(selected['roc']['fpr'], fpr)
    np.testing.assert_allclose(selected['roc']['tpr'], tpr)
    assert f1_score(test.Reaction_Score, predicted) == selected['f1']
    assert selection['selected'] == max(selection['models'], key=lambda n: selection['models'][n]['f1'])


def test_attention_math():
    assert encode_sequence('ACGU').shape == (4, 8)
    X = encode_sequence('ACGUACGUACGGCUA')
    w, o = self_attention(X); tw, to = torch_self_attention(X)
    assert w.shape == (15, 15)
    np.testing.assert_allclose(w.sum(1), 1)
    np.testing.assert_allclose(w, tw)
    np.testing.assert_allclose(o, to)
    assert (w >= 0).all()
    # Repeated bases have identical representations without positional encoding.
    np.testing.assert_allclose(w[0], w[4])


@pytest.mark.parametrize('sequence', ['', 'ACGX', 'acgu', 'A' * 51])
def test_invalid_sequence(sequence):
    with pytest.raises(ValueError): encode_sequence(sequence)


def test_application_length():
    with pytest.raises(ValueError): run('ACGU')


@pytest.mark.parametrize('field,value', [('Eligible_For_Vaccine', 'true'), ('Recommended_Dose_mg', '0.1'), ('Recommended_Dose_mg', -1.), ('Recommended_Dose_mg', float('nan')), ('Reasoning_Steps', []), ('Reasoning_Steps', [3]), ('Disclaimer', 'wrong')])
def test_strict_schema(field, value):
    payload = valid(); payload[field] = value
    with pytest.raises(ValidationError): ClinicalSafetyReport.model_validate(payload)


def test_schema_required_and_extra():
    for key in valid():
        data = valid(); data.pop(key)
        with pytest.raises(ValidationError): ClinicalSafetyReport.model_validate(data)
    with pytest.raises(ValidationError): ClinicalSafetyReport.model_validate(valid() | {'Extra': 1})


def test_retry_once_and_dose_guard(tmp_path):
    responses = iter(['not json', json.dumps(valid())]); calls = []
    def mock(messages):
        calls.append(messages); return next(responses)
    report, elapsed = generate_report(risk(), ['Maximum safe dose is 100 micrograms.'], invoke=mock, output_dir=tmp_path)
    assert len(calls) == 2 and elapsed > 0
    assert not report.Eligible_For_Vaccine and report.Recommended_Dose_mg == 0
    assert any('explicit maximum' in s for s in report.Reasoning_Steps)
    assert len(list(tmp_path.glob('*.md'))) == 1
    assert report.Disclaimer == DISCLAIMER


def test_retry_final_failure(tmp_path):
    calls = []
    def mock(messages): calls.append(messages); return '{}'
    with pytest.raises(ReportGenerationError, match='exactly one retry'):
        generate_report(risk(), ['source'], invoke=mock, output_dir=tmp_path)
    assert len(calls) == 2 and not list(tmp_path.iterdir())


def test_unsupported_citation(tmp_path):
    with pytest.raises(ReportGenerationError, match='verbatim'):
        generate_report(risk(), ['Different source'], invoke=lambda _: json.dumps(valid()), output_dir=tmp_path)


def test_review_threshold_is_not_maximum():
    source = 'Doses above 100 micrograms require secondary clinician review.'
    assert explicit_maximum_mg([source], 58) is None
    report, _ = apply_safety_guards(ClinicalSafetyReport(**valid([source])), risk(), [source])
    assert not report.Eligible_For_Vaccine
    assert any('review threshold' in r for r in report.Reasoning_Steps)


def test_units_and_age_scoping():
    assert explicit_maximum_mg(['Maximum safe dose is 0.1 mg.'], 50) == .1
    assert explicit_maximum_mg(['Ages 5 to 11: maximum dose is 25 mcg.'], 50) is None
    assert explicit_maximum_mg(['Ages 5 to 11: maximum dose is 25 mcg.'], 8) == .025


def test_insufficient_evidence(tmp_path):
    report, _ = generate_report(risk(), [], invoke=lambda _: json.dumps(valid([])), output_dir=tmp_path)
    assert not report.Eligible_For_Vaccine and report.Recommended_Dose_mg == 0
    assert any('No guideline' in x for x in report.Reasoning_Steps)


def test_prompt_grounding():
    prompt = build_prompt(risk(), ['reference text'])
    payload = json.loads(prompt[1][1])
    assert payload == {'risk_output': risk(), 'guideline_excerpts': ['reference text'], 'required_risk_statement': risk_statement(risk())}


def test_persisted_chunks_and_retrieval():
    manifest = read_json(OUTPUT / 'chroma_store/manifest.json')
    source = (DATA / 'mrna_safety_guidelines.txt').read_text()
    for i, c in enumerate(manifest['chunks']):
        assert 300 <= c['token_count'] <= 500
        assert c['text'] == source[c['start_char']:c['end_char']]
        if i: assert manifest['chunks'][i-1]['end_token'] - c['start_token'] == 50
    from src.rag_index import GuidelineIndex, QUESTIONS
    index = GuidelineIndex().load()
    assert index.collection.count() == len(manifest['chunks'])
    stored = index.collection.get(include=['embeddings'])
    assert len(stored['embeddings']) == len(manifest['chunks'])
    for question, expected in QUESTIONS:
        query = index.query(question)
        assert len(query['hits']) == 3
        assert 0 < query['latency_seconds'] < 5
        distances = [h['distance'] for h in query['hits']]
        assert distances == sorted(distances)
        assert any(expected in h['text'] for h in query['hits'])
    with pytest.raises(ValueError): index.query('question', k=index.collection.count()+1)
    with pytest.raises(ValueError): index.query('')


def test_fresh_process_index_reload():
    code = "from src.rag_index import GuidelineIndex; x=GuidelineIndex().load(); r=x.query('May thawed product be refrozen?'); assert len(r['hits']) == 3; print('RELOAD_OK')"
    import os
    process = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120,
                             env=os.environ | {'HF_HUB_OFFLINE': '1'})
    assert process.returncode == 0, process.stderr
    assert 'RELOAD_OK' in process.stdout


def test_report_samples_and_dashboard():
    from PIL import Image
    source = (DATA / 'mrna_safety_guidelines.txt').read_text()
    reports = sorted((OUTPUT / 'reports').glob('*.json'))
    assert len(reports) == 5
    for path in reports:
        report = ClinicalSafetyReport.model_validate(read_json(path))
        assert report.Disclaimer == DISCLAIMER
        assert all(c in source for c in report.FDA_Safety_Warnings_Cited)
        assert report.FDA_Safety_Warnings_Cited
        from src.risk_model import predict_patient
        patient = load_records().set_index('Patient_ID').loc[path.stem].to_dict()
        assert report.Reasoning_Steps[0] == risk_statement(predict_patient(patient))
        assert path.with_suffix('.md').exists()
    timing = read_json(OUTPUT / 'latencies.json')
    assert len(timing['rag_query_seconds']) == len(timing['llm_report_seconds']) == 5
    assert all(t > 0 for t in timing['llm_report_seconds'])
    assert timing['rag_seconds'] == pytest.approx(np.mean(timing['rag_query_seconds']))
    assert timing['report_seconds'] == pytest.approx(np.mean(timing['llm_report_seconds']))
    with Image.open(OUTPUT / 'figures/dashboard.png') as im:
        assert min(im.info['dpi']) >= 150


def test_network_failure_is_not_schema_retry(tmp_path):
    calls = []
    def fail(messages):
        calls.append(messages)
        raise ConnectionError('simulated provider outage')
    with pytest.raises(ConnectionError, match='provider outage'):
        generate_report(risk(), ['source'], invoke=fail, output_dir=tmp_path)
    assert len(calls) == 1


def test_live_key_required(monkeypatch, tmp_path):
    import dotenv
    monkeypatch.delenv('LLM_API_KEY', raising=False)
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *args, **kwargs: None)
    with pytest.raises(ReportGenerationError, match='LLM_API_KEY'):
        generate_report(risk(), ['source'], output_dir=tmp_path)


def test_live_langchain_wiring(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import langchain_openai
    seen = {}
    class MockChat:
        def __init__(self, **kwargs): seen['config'] = kwargs
        def invoke(self, messages):
            seen['messages'] = messages
            return SimpleNamespace(content=json.dumps(valid()))
    monkeypatch.setenv('LLM_API_KEY', 'test-placeholder-not-a-real-key')
    monkeypatch.setenv('LLM_MODEL', 'test-model')
    monkeypatch.setattr(langchain_openai, 'ChatOpenAI', MockChat)
    report, elapsed = generate_report(risk(), ['Maximum safe dose is 100 micrograms.'], output_dir=tmp_path)
    assert seen['config']['max_retries'] == 0
    assert seen['config']['temperature'] == 0
    assert seen['config']['model'] == 'test-model'
    assert elapsed > 0
    assert not report.Eligible_For_Vaccine


def test_report_reruns_replace_same_pair(tmp_path):
    for _ in range(2):
        generate_report(risk(), [], invoke=lambda _: json.dumps(valid([])), output_dir=tmp_path, report_id='sample')
    assert sorted(p.suffix for p in tmp_path.iterdir()) == ['.json', '.md']


def test_excess_dose_veto_removes_contradictory_reasoning():
    payload = valid() | {'Eligible_For_Vaccine': True, 'Reasoning_Steps': ['Eligible at excessive dose.']}
    result, _ = apply_safety_guards(ClinicalSafetyReport(**payload), risk(), ['Maximum safe dose is 100 micrograms.'])
    assert not result.Eligible_For_Vaccine and result.Recommended_Dose_mg == 0
    assert 'Eligible at excessive dose.' not in result.Reasoning_Steps


@pytest.mark.parametrize('age,expected', [(64, None), (65, .05), (80, .05)])
def test_older_age_maximum(age, expected):
    assert explicit_maximum_mg(['Ages 65 and older: maximum safe dose is 50 mcg.'], age) == expected


def test_under_age_maximum():
    assert explicit_maximum_mg(['Age under 5: maximum dose is 25 mcg.'], 4) == .025
    assert explicit_maximum_mg(['Age under 5: maximum dose is 25 mcg.'], 5) is None


def test_risk_statement_validation_retry(tmp_path):
    wrong = valid() | {'Reasoning_Steps': ['Incorrect risk value.']}
    replies = iter([json.dumps(wrong), json.dumps(valid())])
    calls = []
    def mock(messages):
        calls.append(messages)
        return next(replies)
    report, _ = generate_report(risk(), ['Maximum safe dose is 100 micrograms.'], invoke=mock, output_dir=tmp_path)
    assert len(calls) == 2
    assert report.Reasoning_Steps[0] == risk_statement(risk())


def test_saved_attention_recomputes():
    saved = np.load(OUTPUT / 'attention.npz')
    expected, values = self_attention(encode_sequence(str(saved['sequence'])))
    np.testing.assert_allclose(saved['weights'], expected)
    np.testing.assert_allclose(saved['output'], values)


def test_no_clinical_facts_in_system_prompt():
    from src.report_generator import SYSTEM_PROMPT
    assert '100 micrograms' not in SYSTEM_PROMPT
    assert '65' not in SYSTEM_PROMPT
    assert 'no numeric normal' not in SYSTEM_PROMPT


def test_retrieval_verbatim_chunks_and_max_length():
    from src.rag_index import GuidelineIndex, chunk_document
    index = GuidelineIndex().load()
    chunks = chunk_document((DATA / 'mrna_safety_guidelines.txt').read_text(), index.model.tokenizer)
    manifest = read_json(OUTPUT / 'chroma_store/manifest.json')
    assert chunks == manifest['chunks']
    stored = index.collection.get(include=['metadatas'])
    stored_meta = dict(zip(stored['ids'], stored['metadatas']))
    for c in chunks:
        assert stored_meta[c['id']]['token_count'] == c['token_count']
        tokens = index.model.tokenizer.encode(c['text'], add_special_tokens=False)
        assert len(tokens) == c['token_count']
        assert 300 <= len(tokens) <= 500
        assert len(tokens) + 2 <= index.model.max_seq_length


def test_dashboard_has_three_panels_and_uses_artifacts(tmp_path, monkeypatch):
    import shutil
    import matplotlib.pyplot as plt
    from src.common import save_json
    from src.dashboard import build_dashboard
    (tmp_path / 'figures').mkdir()
    for relative in ['figures/model_metrics.json', 'attention.npz', 'latencies.json']:
        shutil.copyfile(OUTPUT / relative, tmp_path / relative)
    figures = []
    real_close = plt.close
    monkeypatch.setattr(plt, 'close', lambda fig=None: figures.append(fig) if hasattr(fig, 'axes') else None)
    build_dashboard(tmp_path)
    fig = figures[-1]
    assert len(fig.axes) == 3
    changed = read_json(tmp_path / 'latencies.json')
    changed['report_seconds'] = 12.345  # Isolated test input, never a project result.
    save_json(tmp_path / 'latencies.json', changed)
    build_dashboard(tmp_path)
    assert figures[-1].axes[2].patches[1].get_height() == 12.345
    for fig in figures: real_close(fig)


def test_both_models_and_reloaded_winner_match_original_run(tmp_path):
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from src.risk_model import train as fit_models
    selected, models = fit_models(output=tmp_path)
    assert set(models) == {'logistic_regression', 'random_forest'}
    raw = load_records()
    train_df, test_df = train_test_split(raw, test_size=.2, stratify=raw.Reaction_Score, random_state=42)
    saved = read_json(OUTPUT / 'figures/model_metrics.json')
    assert selected == saved['selected']
    for name, model in models.items():
        predictions = model.predict(test_df[FEATURES])
        assert classification_report(test_df.Reaction_Score, predictions, output_dict=True, zero_division=0) == saved['models'][name]['classification_report']
        assert confusion_matrix(test_df.Reaction_Score, predictions).tolist() == saved['models'][name]['confusion_matrix']
    reloaded = joblib.load(tmp_path / 'risk_model.joblib')
    np.testing.assert_allclose(reloaded.predict_proba(test_df[FEATURES]), models[selected].predict_proba(test_df[FEATURES]))


def test_clean_csv_and_source_fidelity():
    from src.data_pipeline import clean_records
    from sklearn.model_selection import train_test_split
    raw = load_records()
    train_df, _ = train_test_split(raw, test_size=.2, stratify=raw.Reaction_Score, random_state=42)
    expected = clean_records(raw, train_df)
    saved = pd.read_csv(DATA / 'patient_records_clean.csv', keep_default_na=False, na_values=[''])
    assert not saved.isna().any().any()
    pd.testing.assert_frame_equal(expected, saved)


def test_saved_citations_are_in_patient_retrieval():
    from src.risk_model import predict_patient
    from src.rag_index import GuidelineIndex
    from src.report_generator import patient_query
    index = GuidelineIndex().load()
    for _, row in load_records().head(5).iterrows():
        excerpts = index.query(patient_query(row))['hits']
        report = read_json(OUTPUT / 'reports' / f'{row.Patient_ID}.json')
        assert all(any(quote in hit['text'] for hit in excerpts) for quote in report['FDA_Safety_Warnings_Cited'])


def test_safety_guard_cites_its_retrieved_basis():
    source = (DATA / 'mrna_safety_guidelines.txt').read_text()
    payload = valid([]) | {'Eligible_For_Vaccine': True, 'Recommended_Dose_mg': .05,
        'FDA_Safety_Warnings_Cited': ['Localized injection-site reactions (soreness, redness, mild swelling) are common and do not by themselves require secondary review.']}
    result, _ = apply_safety_guards(ClinicalSafetyReport(**payload), risk(), [source])
    assert not result.Eligible_For_Vaccine
    for prefix in ['Known severe allergic reaction', 'Dose selection for mRNA therapeutic products', 'Patients presenting with an inflammatory biomarker level']:
        assert any(c.startswith(prefix) for c in result.FDA_Safety_Warnings_Cited)
    assert all(c in source for c in result.FDA_Safety_Warnings_Cited)

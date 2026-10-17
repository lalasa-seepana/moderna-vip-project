"""Validate raw records without treating the literal category 'None' as missing."""
from .common import DATA
import numpy as np
import pandas as pd

NUMERIC_COLS = ['Age', 'Biomarker_Level', 'Vaccine_Dose_mcg']
CATEGORICAL_COL = 'Prior_Reaction_History'
FEATURES = NUMERIC_COLS + ['Pre_Existing_Conditions', CATEGORICAL_COL]
TARGET = 'Reaction_Score'
REQUIRED = ['Patient_ID'] + FEATURES + [TARGET]


class DataValidationError(ValueError):
    pass


def load_records(path=DATA / 'patient_records.csv'):
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise DataValidationError('Missing required columns: ' + ', '.join(missing))
    if df.empty:
        raise DataValidationError('Dataset is empty')
    if df.Patient_ID.isna().any() or df.Patient_ID.duplicated().any():
        raise DataValidationError('Patient_ID must be present and unique')
    for col in NUMERIC_COLS + ['Pre_Existing_Conditions', TARGET]:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
        except (ValueError, TypeError) as exc:
            raise DataValidationError(f'{col} must be numeric') from exc
        values = df[col].dropna()
        if not np.isfinite(values).all():
            raise DataValidationError(f'{col} contains non-finite values')
    for col in ['Age', 'Vaccine_Dose_mcg', 'Pre_Existing_Conditions', TARGET]:
        if df[col].isna().any():
            raise DataValidationError(f'{col} contains missing values')
    for col in ['Biomarker_Level', CATEGORICAL_COL]:
        if df[col].isna().all():
            raise DataValidationError(f'{col} is entirely missing; cannot impute')
    if not df[CATEGORICAL_COL].dropna().isin(['None', 'Mild', 'Moderate']).all():
        raise DataValidationError('Prior_Reaction_History has an unknown category')
    if not df[TARGET].isin([0, 1]).all():
        raise DataValidationError('Reaction_Score must contain only 0 or 1')
    for col in ['Age', 'Pre_Existing_Conditions']:
        if (df[col] % 1 != 0).any():
            raise DataValidationError(f'{col} must be an integer')
    if not df.Age.between(0, 120).all():
        raise DataValidationError('Age must be between 0 and 120')
    for col in ['Biomarker_Level', 'Pre_Existing_Conditions']:
        if (df[col].dropna() < 0).any():
            raise DataValidationError(f'{col} cannot be negative')
    if (df.Vaccine_Dose_mcg <= 0).any():
        raise DataValidationError('Vaccine_Dose_mcg must be positive')
    return df


def clean_records(df, reference):
    """Imputation statistics come from training rows only."""
    result = df.copy()
    result['Biomarker_Level'] = result.Biomarker_Level.fillna(reference.Biomarker_Level.median())
    result[CATEGORICAL_COL] = result[CATEGORICAL_COL].fillna(reference[CATEGORICAL_COL].mode().iloc[0])
    return result


def load_and_clean(path=DATA / 'patient_records.csv'):
    """Standalone cleaning helper; model training instead splits before imputation."""
    df = load_records(path)
    return clean_records(df, df)

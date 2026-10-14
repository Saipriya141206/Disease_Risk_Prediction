from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from health.diseases import DISEASES, TARGET_FEATURE_EXCLUSIONS


FEATURES = [
    'BMI', 'Smoking', 'Stroke', 'PhysicalHealth', 'MentalHealth',
    'DiffWalking', 'Sex', 'AgeCategory', 'Diabetic', 'PhysicalActivity', 'GenHealth',
    'SleepTime', 'Asthma', 'KidneyDisease', 'SkinCancer',
]
CATEGORICAL = [
    'Smoking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory',
    'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer',
]
NUMERIC = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
CHUNK_SIZE = 50_000
SAMPLES_PER_CHUNK = 3_000
MAX_ROWS_PER_SOURCE = 15_000


def _column(frame, *names):
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(np.nan, index=frame.index)


def _yes_no(values):
    normalized = values.astype('string').str.strip().str.lower()
    return normalized.map({
        'yes': 1.0, 'yes.': 1.0, '1': 1.0, '1.0': 1.0,
        'no': 0.0, 'no.': 0.0, '2': 0.0, '2.0': 0.0,
    })


def _missing_target(frame):
    return pd.Series(np.nan, index=frame.index)


def _combined_target(first, second):
    target = pd.Series(np.nan, index=first.index)
    target[(first == 1) | (second == 1)] = 1
    target[(first == 0) & (second == 0)] = 0
    return target


def _diabetes_category(values, *, coded=False):
    if coded:
        codes = pd.to_numeric(values, errors='coerce')
        return codes.map({1: 'Yes', 2: 'Yes', 3: 'No', 4: 'No, borderline diabetes'})
    diabetes = values.astype('string').str.lower()
    labels = np.select(
        [
            diabetes.str.contains('pre-diabetes|borderline', na=False),
            diabetes.str.startswith('yes', na=False),
            diabetes.str.startswith('no', na=False),
        ],
        ['No, borderline diabetes', 'Yes', 'No'],
        default='',
    )
    return pd.Series(labels, index=values.index).replace('', pd.NA)


def _diabetes_target(values, *, coded=False):
    return _diabetes_category(values, coded=coded).map({'Yes': 1.0, 'No': 0.0, 'No, borderline diabetes': 0.0})


def _age_category(values, *, coded=False):
    if coded:
        codes = pd.to_numeric(values, errors='coerce')
        labels = {
            1: '18-24', 2: '25-29', 3: '30-34', 4: '35-39',
            5: '40-44', 6: '45-49', 7: '50-54', 8: '55-59',
            9: '60-64', 10: '65-69', 11: '70-74', 12: '75-79',
            13: '80 or older', 14: '80 or older',
        }
        return codes.map(labels)

    labels = values.astype('string').str.strip()
    labels = labels.str.replace(r'^Age\s+(\d+)\s+to\s+(\d+)$', r'\1-\2', regex=True)
    labels = labels.str.replace(r'^Age\s+80\s+or\s+older$', '80 or older', regex=True)
    labels = labels.str.replace(r'^80\+$', '80 or older', regex=True)
    return labels.where(labels.isin({
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older',
    }))


def _health_days(values, *, coded=False):
    numbers = pd.to_numeric(values, errors='coerce')
    if coded:
        numbers = numbers.mask(numbers == 88, 0)
    return numbers.where(numbers.between(0, 30))


def _raw_bmi(frame):
    direct = pd.to_numeric(_column(frame, '_BMI5'), errors='coerce') / 100
    weight = pd.to_numeric(_column(frame, 'WEIGHT2'), errors='coerce')
    height = pd.to_numeric(_column(frame, 'HEIGHT3'), errors='coerce')
    feet = np.floor(height / 100)
    inches = feet * 12 + (height % 100)
    derived = 703 * weight / inches.pow(2)
    bmi = direct.where(direct.between(10, 80), derived)
    return bmi.where(bmi.between(10, 80))


def _prepare_chunk(frame, source):
    features = pd.DataFrame(index=frame.index)
    targets = {key: _missing_target(frame) for key, _ in DISEASES}

    if source == '2020':
        targets.update({
            'heart_disease': _yes_no(_column(frame, 'HeartDisease')),
            'stroke': _yes_no(_column(frame, 'Stroke')),
            'diabetes': _diabetes_target(_column(frame, 'Diabetic')),
            'asthma': _yes_no(_column(frame, 'Asthma')),
            'skin_cancer': _yes_no(_column(frame, 'SkinCancer')),
            'kidney_disease': _yes_no(_column(frame, 'KidneyDisease')),
        })
        features['BMI'] = pd.to_numeric(_column(frame, 'BMI'), errors='coerce')
        features['Smoking'] = _yes_no(_column(frame, 'Smoking'))
        features['Stroke'] = _yes_no(_column(frame, 'Stroke'))
        features['Sex'] = _column(frame, 'Sex')
        features['AgeCategory'] = _age_category(_column(frame, 'AgeCategory'))
        features['Diabetic'] = _diabetes_category(_column(frame, 'Diabetic'))
        features['PhysicalActivity'] = _yes_no(_column(frame, 'PhysicalActivity'))
        features['GenHealth'] = _column(frame, 'GenHealth')
        features['PhysicalHealth'] = _health_days(_column(frame, 'PhysicalHealth'))
        features['MentalHealth'] = _health_days(_column(frame, 'MentalHealth'))
        features['DiffWalking'] = _yes_no(_column(frame, 'DiffWalking'))
        features['SleepTime'] = pd.to_numeric(_column(frame, 'SleepTime'), errors='coerce')
        features['Asthma'] = _yes_no(_column(frame, 'Asthma'))
        features['KidneyDisease'] = _yes_no(_column(frame, 'KidneyDisease'))
        features['SkinCancer'] = _yes_no(_column(frame, 'SkinCancer'))
    elif source == '2022':
        heart_attack = _yes_no(_column(frame, 'HadHeartAttack'))
        angina = _yes_no(_column(frame, 'HadAngina'))
        diabetes_values = _column(frame, 'HadDiabetes')
        targets.update({
            'heart_disease': _combined_target(heart_attack, angina),
            'heart_attack': heart_attack,
            'angina': angina,
            'stroke': _yes_no(_column(frame, 'HadStroke')),
            'diabetes': _diabetes_target(diabetes_values),
            'asthma': _yes_no(_column(frame, 'HadAsthma')),
            'skin_cancer': _yes_no(_column(frame, 'HadSkinCancer')),
            'kidney_disease': _yes_no(_column(frame, 'HadKidneyDisease')),
            'copd': _yes_no(_column(frame, 'HadCOPD')),
            'arthritis': _yes_no(_column(frame, 'HadArthritis')),
            'depression': _yes_no(_column(frame, 'HadDepressiveDisorder')),
        })
        features['BMI'] = pd.to_numeric(_column(frame, 'BMI'), errors='coerce')
        smoking = _column(frame, 'SmokerStatus').astype('string').str.lower()
        features['Smoking'] = np.select(
            [smoking.str.startswith('never smoked', na=False), smoking.str.contains('smok', na=False)],
            [0.0, 1.0], default=np.nan,
        )
        features['Stroke'] = _yes_no(_column(frame, 'HadStroke'))
        features['Sex'] = _column(frame, 'Sex')
        features['AgeCategory'] = _age_category(_column(frame, 'AgeCategory'))
        features['Diabetic'] = _diabetes_category(diabetes_values)
        features['PhysicalActivity'] = _yes_no(_column(frame, 'PhysicalActivities'))
        features['GenHealth'] = _column(frame, 'GeneralHealth')
        features['PhysicalHealth'] = _health_days(_column(frame, 'PhysicalHealthDays'))
        features['MentalHealth'] = _health_days(_column(frame, 'MentalHealthDays'))
        features['DiffWalking'] = _yes_no(_column(frame, 'DifficultyWalking'))
        features['SleepTime'] = pd.to_numeric(_column(frame, 'SleepHours'), errors='coerce')
        features['Asthma'] = _yes_no(_column(frame, 'HadAsthma'))
        features['KidneyDisease'] = _yes_no(_column(frame, 'HadKidneyDisease'))
        features['SkinCancer'] = _yes_no(_column(frame, 'HadSkinCancer'))
    else:
        heart_attack = _yes_no(_column(frame, 'CVDINFR4'))
        coronary_disease = _yes_no(_column(frame, 'CVDCRHD4'))
        diabetes_values = _column(frame, 'DIABETE3')
        blood_pressure = pd.to_numeric(_column(frame, 'BPHIGH4'), errors='coerce')
        targets.update({
            'heart_disease': _combined_target(heart_attack, coronary_disease),
            'heart_attack': heart_attack,
            'angina': coronary_disease,
            'stroke': _yes_no(_column(frame, 'CVDSTRK3')),
            'diabetes': _diabetes_target(diabetes_values, coded=True),
            'asthma': _yes_no(_column(frame, 'ASTHMA3')),
            'skin_cancer': _yes_no(_column(frame, 'CHCSCNCR')),
            'other_cancer': _yes_no(_column(frame, 'CHCOCNCR')),
            'kidney_disease': _yes_no(_column(frame, 'CHCKIDNY')),
            'copd': _yes_no(_column(frame, 'CHCCOPD', 'CHCCOPD1')),
            'arthritis': _yes_no(_column(frame, 'HAVARTH3')),
            'depression': _yes_no(_column(frame, 'ADDEPEV2')),
            'high_blood_pressure': blood_pressure.map({1: 1.0, 2: 1.0, 3: 0.0, 4: 0.0}),
        })

        features['BMI'] = _raw_bmi(frame)
        features['Smoking'] = _yes_no(_column(frame, 'SMOKE100'))
        features['Stroke'] = _yes_no(_column(frame, 'CVDSTRK3'))
        sex = _column(frame, 'SEX').astype('string').str.strip()
        features['Sex'] = sex.map({'1.0': 'Male', '1': 'Male', '2.0': 'Female', '2': 'Female'})
        age_group = _column(frame, '_AGEG5YR')
        if age_group.notna().any():
            features['AgeCategory'] = _age_category(age_group, coded=True)
        else:
            age = pd.to_numeric(_column(frame, 'AGE', '_AGE80'), errors='coerce')
            features['AgeCategory'] = pd.cut(
                age, bins=[17, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, np.inf],
                labels=['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'],
            )
        features['Diabetic'] = _diabetes_category(diabetes_values, coded=True)
        features['PhysicalActivity'] = _yes_no(_column(frame, 'EXERANY2'))
        general_health = _column(frame, 'GENHLTH').astype('string').str.strip()
        features['GenHealth'] = general_health.map({
            '1': 'Excellent', '1.0': 'Excellent', '2': 'Very good', '2.0': 'Very good',
            '3': 'Good', '3.0': 'Good', '4': 'Fair', '4.0': 'Fair', '5': 'Poor', '5.0': 'Poor',
        })
        features['PhysicalHealth'] = _health_days(_column(frame, 'PHYSHLTH'), coded=True)
        features['MentalHealth'] = _health_days(_column(frame, 'MENTHLTH'), coded=True)
        features['DiffWalking'] = _yes_no(_column(frame, 'DIFFWALK'))
        features['SleepTime'] = pd.to_numeric(_column(frame, 'SLEPTIM1'), errors='coerce')
        features['Asthma'] = _yes_no(_column(frame, 'ASTHMA3'))
        features['KidneyDisease'] = _yes_no(_column(frame, 'CHCKIDNY'))
        features['SkinCancer'] = _yes_no(_column(frame, 'CHCSCNCR'))

    features['BMI'] = pd.to_numeric(features['BMI'], errors='coerce').where(
        pd.to_numeric(features['BMI'], errors='coerce').between(10, 80),
    )
    for column in ('SleepTime',):
        features[column] = pd.to_numeric(features[column], errors='coerce').where(
            pd.to_numeric(features[column], errors='coerce').between(1, 24),
        )
    for column in CATEGORICAL:
        if column not in ('Sex', 'AgeCategory', 'Diabetic', 'GenHealth'):
            features[column] = pd.to_numeric(features[column], errors='coerce').map({0.0: 'No', 1.0: 'Yes'})
    features['Sex'] = features['Sex'].astype('string').where(features['Sex'].isin(['Female', 'Male']))
    features['Diabetic'] = features['Diabetic'].astype('string').where(features['Diabetic'].isin([
        'No', 'Yes', 'No, borderline diabetes',
    ]))
    features['GenHealth'] = features['GenHealth'].astype('string').where(features['GenHealth'].isin([
        'Excellent', 'Very good', 'Good', 'Fair', 'Poor',
    ]))
    features['AgeCategory'] = features['AgeCategory'].astype('string')
    for column in CATEGORICAL:
        values = features[column].astype(object)
        features[column] = values.where(pd.notna(values), np.nan)
    for column in NUMERIC:
        features[column] = pd.to_numeric(features[column], errors='coerce').astype('float64')
    for key, target in targets.items():
        features[f'target_{key}'] = pd.to_numeric(target, errors='coerce').astype('float64')
    target_columns = [f'target_{key}' for key, _ in DISEASES]
    return features.dropna(subset=target_columns, how='all')


def _source_columns(source):
    if source == '2020':
        return FEATURES + ['HeartDisease']
    if source == '2022':
        return [
            'BMI', 'SmokerStatus', 'HadStroke', 'Sex', 'AgeCategory', 'HadDiabetes',
            'PhysicalActivities', 'GeneralHealth', 'PhysicalHealthDays', 'MentalHealthDays',
            'DifficultyWalking', 'SleepHours', 'HadAsthma', 'HadKidneyDisease',
            'HadSkinCancer', 'HadHeartAttack', 'HadAngina', 'HadCOPD', 'HadArthritis',
            'HadDepressiveDisorder',
        ]
    return [
        'CVDINFR4', 'CVDCRHD4', 'WEIGHT2', 'HEIGHT3', '_BMI5', 'SMOKE100', 'CVDSTRK3',
        'SEX', '_AGEG5YR', 'AGE', '_AGE80', 'DIABETE3', 'EXERANY2', 'GENHLTH',
        'PHYSHLTH', 'MENTHLTH', 'DIFFWALK', 'SLEPTIM1', 'ASTHMA3', 'CHCKIDNY', 'CHCSCNCR',
        'CHCOCNCR', 'CHCCOPD', 'CHCCOPD1', 'HAVARTH3', 'ADDEPEV2', 'BPHIGH4',
    ]


def _read_source(path, source):
    header = pd.read_csv(path, nrows=0).columns
    use_columns = [column for column in _source_columns(source) if column in header]
    samples = []
    target_columns = {key: f'target_{key}' for key, _ in DISEASES}
    valid_rows = {key: 0 for key, _ in DISEASES}
    for chunk_number, chunk in enumerate(pd.read_csv(
        path, usecols=use_columns, chunksize=CHUNK_SIZE, low_memory=False,
    )):
        prepared = _prepare_chunk(chunk, source)
        for key, column in target_columns.items():
            valid_rows[key] += int(prepared[column].notna().sum())
        if len(prepared) > SAMPLES_PER_CHUNK:
            prepared = prepared.sample(n=SAMPLES_PER_CHUNK, random_state=42 + chunk_number)
        samples.append(prepared)
    if not samples:
        raise CommandError(f'No usable rows found in {path.name}.')
    result = pd.concat(samples, ignore_index=True)
    if len(result) > MAX_ROWS_PER_SOURCE:
        result = result.sample(n=MAX_ROWS_PER_SOURCE, random_state=42)
    return result.reset_index(drop=True), valid_rows


class Command(BaseCommand):
    help = 'Train disease-specific Random Forest models from the uploaded BRFSS CSVs.'

    def add_arguments(self, parser):
        parser.add_argument('--data-dir', default=str(Path(settings.BASE_DIR) / 'data'))

    def handle(self, *args, **options):
        data_dir = Path(options['data_dir'])
        if not data_dir.is_dir():
            raise CommandError(f'Dataset directory not found: {data_dir}')
        sources = [
            *((data_dir / f'{year}.csv', 'raw') for year in range(2011, 2016)),
            (data_dir / 'heart_2020_cleaned.csv', '2020'),
            (data_dir / 'heart_2022_with_nans.csv', '2022'),
        ]
        missing_files = [path.name for path, _ in sources if not path.is_file()]
        if missing_files:
            raise CommandError(f'Missing expected CSV files: {", ".join(missing_files)}')

        frames = {}
        row_counts = {}
        for path, source in sources:
            self.stdout.write(f'Reading {path.name}...')
            frames[path.name], row_counts[path.name] = _read_source(path, source)
            self.stdout.write(f'  Sampled rows: {len(frames[path.name]):,}')

        all_data = pd.concat(frames.values(), ignore_index=True)
        holdout_by_disease = {
            'other_cancer': '2015.csv',
            'high_blood_pressure': '2015.csv',
        }
        models = {}
        metrics = {}
        for key, label in DISEASES:
            target_column = f'target_{key}'
            holdout_name = holdout_by_disease.get(key, 'heart_2022_with_nans.csv')
            validation_rows = frames[holdout_name].dropna(subset=[target_column])
            training_rows = pd.concat(
                [frame for name, frame in frames.items() if name != holdout_name],
                ignore_index=True,
            ).dropna(subset=[target_column])
            if training_rows[target_column].nunique() != 2 or validation_rows[target_column].nunique() != 2:
                self.stderr.write(self.style.WARNING(f'Skipping {label}: not enough labeled classes in train/holdout data.'))
                continue

            feature_columns = [
                feature for feature in FEATURES
                if feature != TARGET_FEATURE_EXCLUSIONS.get(key)
            ]
            train_x = training_rows[feature_columns]
            train_y = training_rows[target_column].astype(int)
            test_x = validation_rows[feature_columns]
            test_y = validation_rows[target_column].astype(int)
            preprocess = ColumnTransformer([
                ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median'))]), NUMERIC),
                ('categorical', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encode', OneHotEncoder(handle_unknown='ignore')),
                ]), [name for name in CATEGORICAL if name in feature_columns]),
            ])
            model = Pipeline([
                ('preprocess', preprocess),
                ('classifier', RandomForestClassifier(
                    n_estimators=80,
                    max_features='sqrt',
                    min_samples_leaf=4,
                    class_weight='balanced_subsample',
                    n_jobs=-1,
                    random_state=42,
                )),
            ])
            model.fit(train_x, train_y)
            positive_class = list(model.classes_).index(1)
            probabilities = model.predict_proba(test_x)[:, positive_class]
            predictions = model.predict(test_x)
            disease_metrics = {
                'label': label,
                'validation_source': holdout_name,
                'train_rows': int(len(training_rows)),
                'validation_rows': int(len(validation_rows)),
                'accuracy': float(accuracy_score(test_y, predictions)),
                'roc_auc': float(roc_auc_score(test_y, probabilities)),
                'classification_report': classification_report(
                    test_y, predictions, target_names=['No', 'Yes'],
                    output_dict=True, zero_division=0,
                ),
                'confusion_matrix': confusion_matrix(test_y, predictions).tolist(),
                'source_valid_rows': {name: values[key] for name, values in row_counts.items()},
            }
            metrics[key] = disease_metrics

            final_rows = all_data.dropna(subset=[target_column])
            model.fit(final_rows[feature_columns], final_rows[target_column].astype(int))
            models[key] = {'label': label, 'features': feature_columns, 'pipeline': model}
            self.stdout.write(
                f'{label}: holdout {holdout_name}, n={len(validation_rows):,}, '
                f'accuracy={disease_metrics["accuracy"]:.3f}, AUC={disease_metrics["roc_auc"]:.3f}'
            )

        if not models:
            raise CommandError('No disease models could be trained from these files.')
        output_path = Path(settings.BASE_DIR) / 'models' / 'heart_risk.joblib'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model_type': 'multi_disease_random_forest', 'models': models}, output_path)
        metrics_path = output_path.with_name('heart_risk_metrics.json')
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        self.stdout.write(self.style.SUCCESS(
            f'Trained {len(models)} disease-specific Random Forest models.\n'
            f'Model bundle saved to {output_path}\nMetrics saved to {metrics_path}'
        ))
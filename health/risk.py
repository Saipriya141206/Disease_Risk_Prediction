from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from django.conf import settings

from .diseases import DISEASES


MODEL_PATH = Path(settings.BASE_DIR) / 'models' / 'heart_risk.joblib'


@lru_cache(maxsize=1)
def _load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def _age_category(age):
    if age <= 24:
        return '18-24'
    bands = [24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79]
    for upper in bands:
        if age <= upper:
            return f'{upper - 4}-{upper}'
    return '80 or older'


def _model_features(profile):
    return {
        'BMI': profile.bmi,
        'Smoking': 'Yes' if profile.smoking else 'No',
        'Stroke': 'Yes' if profile.stroke else 'No',
        'PhysicalHealth': profile.physical_health_days,
        'MentalHealth': profile.mental_health_days,
        'DiffWalking': 'Yes' if profile.difficulty_walking else 'No',
        'Sex': profile.sex,
        'AgeCategory': _age_category(profile.age),
        'Diabetic': profile.diabetic,
        'PhysicalActivity': 'Yes' if profile.physical_activity else 'No',
        'GenHealth': profile.general_health,
        'SleepTime': profile.sleep_hours,
        'Asthma': 'Yes' if profile.asthma else 'No',
        'KidneyDisease': 'Yes' if profile.kidney_disease else 'No',
        'SkinCancer': 'Yes' if profile.skin_cancer else 'No',
    }


def predict_disease_risks(profile):
    model_bundle = _load_model() if MODEL_PATH.exists() else None
    if model_bundle is None:
        return []

    profile_features = _model_features(profile)
    predictions = []
    if isinstance(model_bundle, dict):
        disease_models = model_bundle.get('models', {})
        for key, label in DISEASES:
            model_entry = disease_models.get(key)
            if not model_entry:
                continue
            model = model_entry['pipeline']
            feature_names = model_entry['features']
            features = pd.DataFrame([{
                name: profile_features[name] for name in feature_names
            }])
            classes = list(model.classes_)
            probability = float(model.predict_proba(features)[0][classes.index(1)])
            score = round(probability * 100, 1)
            predictions.append({
                'key': key,
                'label': label,
                'probability_pct': score,
                'risk_level': _risk_level(score),
            })
        return predictions

    probability = float(model_bundle.predict_proba(pd.DataFrame([profile_features]))[0][1])
    score = round(probability * 100, 1)
    return [{
        'key': 'heart_disease',
        'label': 'Heart disease',
        'probability_pct': score,
        'risk_level': _risk_level(score),
    }]


def assess_risk(profile):
    heart_prediction = next(
        (item for item in predict_disease_risks(profile) if item['key'] == 'heart_disease'),
        None,
    )
    if heart_prediction is not None:
        score = round(heart_prediction['probability_pct'])
        return score, _risk_level(score), 'model'

    score = 5
    score += 18 if profile.smoking else 0
    score += 5 if profile.alcohol_drinking else 0
    score += 20 if profile.stroke else 0
    score += 10 if profile.difficulty_walking else 0
    score += 12 if profile.diabetic != 'No' else 0
    score += 8 if not profile.physical_activity else 0
    score += 12 if profile.general_health == 'Poor' else 6 if profile.general_health == 'Fair' else 0
    score += 10 if profile.bmi >= 30 else 5 if profile.bmi >= 25 else 0
    score += 7 if profile.physical_health_days >= 14 else 4 if profile.physical_health_days >= 5 else 0
    score += 4 if profile.sleep_hours < 6 or profile.sleep_hours > 9 else 0
    score += 10 if profile.family_heart_history else 0
    score += 8 if profile.diet_quality == 'poor' else 3 if profile.diet_quality == 'fair' else 0
    score += 4 if profile.kidney_disease else 0
    score = min(score, 95)
    return score, _risk_level(score), 'screening'


def _risk_level(score):
    if score < 30:
        return 'low'
    if score < 60:
        return 'moderate'
    return 'high'


def recommendations(profile):
    items = []
    if profile.smoking:
        items.append(('Make a quit plan', 'A clinician or local quitline can help you choose support and a realistic next step.', 'Daily habits'))
    if not profile.physical_activity:
        items.append(('Add gentle movement', 'Try a short, comfortable walk or another low-impact activity, building up at a pace that feels safe.', 'Movement'))
    if profile.bmi >= 25:
        items.append(('Build a balanced plate', 'Include vegetables, fruit, whole grains, and satisfying protein. A registered dietitian can help personalize this.', 'Nutrition'))
    if profile.diet_quality != 'good':
        items.append(('Choose one food upgrade', 'Add a fruit, vegetable, or high-fibre option to one meal this week; small repeatable changes count.', 'Nutrition'))
    if profile.sleep_hours < 7 or profile.sleep_hours > 9:
        items.append(('Protect your sleep routine', 'A steady wind-down and wake time can support sleep. Discuss persistent sleep concerns with a healthcare professional.', 'Recovery'))
    if profile.physical_health_days >= 5 or profile.general_health in ('Fair', 'Poor'):
        items.append(('Check in with your care team', 'Share ongoing symptoms and this screening summary with a qualified healthcare professional.', 'Care'))
    if profile.family_heart_history or profile.diabetic != 'No' or profile.stroke:
        items.append(('Review personal risk factors', 'Ask your clinician how family history and medical conditions affect your prevention plan.', 'Care'))
    if not items:
        items.extend([
            ('Keep movement consistent', 'Choose activities you enjoy and build a routine that fits your week.', 'Movement'),
            ('Keep routine checkups', 'Regular preventive visits help you and your care team track the bigger picture.', 'Care'),
        ])
    return items[:4]
DISEASES = [
    ('heart_disease', 'Heart disease'),
    ('heart_attack', 'Heart attack'),
    ('angina', 'Angina / coronary disease'),
    ('stroke', 'Stroke'),
    ('diabetes', 'Diabetes'),
    ('asthma', 'Asthma'),
    ('skin_cancer', 'Skin cancer'),
    ('other_cancer', 'Other cancer'),
    ('kidney_disease', 'Kidney disease'),
    ('copd', 'COPD'),
    ('arthritis', 'Arthritis'),
    ('depression', 'Depression'),
    ('high_blood_pressure', 'High blood pressure'),
]

TARGET_FEATURE_EXCLUSIONS = {
    'stroke': 'Stroke',
    'diabetes': 'Diabetic',
    'asthma': 'Asthma',
    'skin_cancer': 'SkinCancer',
    'kidney_disease': 'KidneyDisease',
}
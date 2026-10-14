# SwasthAI

SwasthAI is a Django prototype for personal health check-ins. A user can record health and lifestyle factors, review condition estimates based on public survey data, get practical wellness suggestions, and revisit prior check-ins. It is an educational prototype, not a clinical product.

## What the application does

- Provides local account creation, sign-in, and sign-out using Django authentication.
- Creates a health profile for each account and lets the user update it through a validated form.
- Saves an assessment record each time the profile is submitted, including an overall score, a low/moderate/high category, the scoring method, and the timestamp.
- Shows a dashboard with available condition estimates, a profile summary, tailored next-step suggestions, and up to six recent check-ins.
- Uses trained disease-specific Random Forest models when `models/heart_risk.joblib` is available. Without that bundle, it saves a hand-weighted screening index for the check-in; the dashboard will show that disease models are unavailable rather than presenting the index as a disease prediction.
- Includes Django admin views for health profiles and assessment records.

### User journey

1. Create an account at `/signup/` or sign in at `/login/`.
2. Complete or update the health profile at `/assessment/`.
3. Review the dashboard at `/`, including condition estimates if the model bundle is installed.
4. Return to the profile to create another timestamped check-in.

The profile collects age, sex, BMI, smoking, alcohol use, stroke history, physical and mental health days, walking difficulty, diabetes status, activity, general health, sleep, asthma, kidney disease, skin cancer, diet quality, and family history of heart disease.

## Project layout

| Path | Purpose |
| --- | --- |
| `config/` | Django settings and root URL configuration |
| `health/models.py` | Health profile and assessment database models |
| `health/forms.py` | Profile form and range validation |
| `health/views.py` and `health/urls.py` | Authentication, dashboard, and assessment flows |
| `health/risk.py` | Model inference, fallback scoring, and recommendations |
| `health/diseases.py` | Supported condition names and target-feature exclusions |
| `health/management/commands/train_risk_model.py` | Dataset preparation, model training, and evaluation |
| `health/templates/health/` | Django templates for authentication, assessment, and dashboard |
| `static/health/css/app.css` | Responsive application styling |
| `data/` | Source survey datasets used by the training command |
| `models/heart_risk.joblib` | Trained model bundle used for inference |
| `models/heart_risk_metrics.json` | Per-condition validation metrics from model training |
| `db.sqlite3` | Local development database |

## Risk estimates and recommendations

The model bundle can contain separate classifiers for heart disease, heart attack, angina/coronary disease, stroke, diabetes, asthma, skin cancer, other cancer, kidney disease, COPD, arthritis, depression, and high blood pressure. The dashboard displays estimates for the models present in the bundle. Each model score is the classifier's positive-class output, shown as a percentage and grouped as low (under 30), moderate (30 through 59), or high (60 and above).

If the heart-disease model is unavailable, `health/risk.py` computes a hand-weighted screening index from selected profile fields. The result is capped at 95, classified using the same bands, and recorded with method `screening` rather than `model`. This index is a product-flow fallback, not a validated score. The dashboard history shows recorded overall assessment scores.

Recommendations are simple rules based on the profile, such as suggesting a care-team discussion, movement, sleep routine, or nutrition change. They are general educational prompts, not individualized medical advice.

## Model training

The training command expects these CSVs under `data/`:

- `2011.csv` through `2015.csv`: annual raw BRFSS extracts.
- `heart_2020_cleaned.csv`: the 2020 Personal Key Indicators of Heart Disease dataset.
- `heart_2022_with_nans.csv`: the 2022 BRFSS extract retaining partially answered records.

Run:

```powershell
python manage.py train_risk_model
```

The command reads only relevant columns in 50,000-row chunks, normalizes year-specific survey codes and labels into common features, samples up to 3,000 rows per chunk, then caps each source at 15,000 sampled rows. It imputes missing numeric and categorical predictors, one-hot encodes categorical predictors, and trains a separate Random Forest for each condition with enough labeled examples.

For conditions represented in 2022, `heart_2022_with_nans.csv` is used as the validation holdout. Other cancer and high blood pressure use `2015.csv` as the holdout because those targets are not available in the 2022 source. Each condition's reported diagnosis is excluded from its predictors where that diagnosis is among the shared input features. After holdout evaluation, each final model is refit using all sampled labeled records. The 2022 complete-case alternate, `heart_2022_no_nans.csv`, is not used, avoiding overlap with the partially answered 2022 cohort.

The command writes the model bundle to `models/heart_risk.joblib` and its validation results to `models/heart_risk_metrics.json`. Metrics include accuracy, ROC AUC, per-class precision/recall/F1, confusion matrix, row counts, and validation source. Check positive-class precision and recall as well as aggregate accuracy: survey conditions can be imbalanced, so accuracy alone can hide weak detection of positive cases. The checked-in metrics file currently reports heart-disease holdout accuracy of about 0.826 and ROC AUC of about 0.816 on 14,844 2022 validation rows; these figures are dataset evaluation results, not clinical performance guarantees.

The data describes survey respondents' reported current or past conditions. The models learn to classify those labels from cross-sectional survey features. Their outputs are not calibrated future disease probabilities, personal prognoses, diagnoses, or clinical decision support. Model performance, data compatibility, and licensing should be reviewed before reuse.

## Local setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/`, create an account, and complete the profile. The project uses Django, SQLite, pandas, scikit-learn, and joblib. If the model artifact is absent, the web application can still run using the screening fallback. To check the application and data-preparation tests, run:

```powershell
python manage.py test
```

## Data, privacy, and safety

The development setup stores user accounts, profile details, and assessment history in local SQLite. Do not use real patient data or deploy this prototype without security, privacy, and clinical review. Before production deployment, configure a secret `DJANGO_SECRET_KEY` and appropriate `DJANGO_ALLOWED_HOSTS`, disable debug mode, use HTTPS, and add suitable access controls, encryption, retention policies, and operational safeguards.

The training inputs are based on CDC Behavioral Risk Factor Surveillance System (BRFSS) data and the 2020 Personal Key Indicators of Heart Disease dataset. Confirm dataset provenance, terms of use, licensing, and feature compatibility before collecting, redistributing, or retraining with data. The project also contains `2015_formats.json`, which documents source-specific formats.

SwasthAI provides educational wellness information only. It does not diagnose disease, replace a clinician, or provide emergency guidance. A user should discuss health concerns and results with a qualified healthcare professional.
from django.contrib.auth.models import User
from django.test import TestCase
from django.urls import reverse
from unittest.mock import patch

from .models import HealthProfile, RiskAssessment
from .risk import _age_category, assess_risk, predict_disease_risks
from .management.commands.train_risk_model import _prepare_chunk
import pandas as pd


class AssessmentFlowTests(TestCase):
	def setUp(self):
		self.user = User.objects.create_user(username='sam', password='a-strong-test-password')
		self.profile = HealthProfile.objects.create(user=self.user)

	def test_anonymous_dashboard_redirects_to_login(self):
		response = self.client.get(reverse('dashboard'))
		self.assertRedirects(response, f'{reverse("login")}?next={reverse("dashboard")}')

	def test_screening_estimate_is_bounded_and_disclosed(self):
		self.profile.smoking = True
		self.profile.diabetic = 'Yes'
		self.profile.save()

		with patch('health.risk._load_model', return_value=None):
			score, level, method = assess_risk(self.profile)

		self.assertGreaterEqual(score, 0)
		self.assertLessEqual(score, 100)
		self.assertEqual(level, 'moderate')
		self.assertEqual(method, 'screening')

	def test_assessment_post_saves_profile_and_history(self):
		self.client.force_login(self.user)
		response = self.client.post(reverse('assessment'), {
			'age': 38,
			'sex': 'Female',
			'bmi': 23.5,
			'smoking': '',
			'alcohol_drinking': '',
			'stroke': '',
			'physical_health_days': 0,
			'mental_health_days': 0,
			'difficulty_walking': '',
			'diabetic': 'No',
			'physical_activity': 'on',
			'general_health': 'Good',
			'sleep_hours': 7,
			'asthma': '',
			'kidney_disease': '',
			'skin_cancer': '',
			'diet_quality': 'good',
			'family_heart_history': '',
		})

		self.assertRedirects(response, reverse('dashboard'))
		self.assertEqual(RiskAssessment.objects.filter(profile=self.profile).count(), 1)

		dashboard_response = self.client.get(reverse('dashboard'))
		self.assertEqual(dashboard_response.status_code, 200)
		self.assertContains(dashboard_response, 'Personal next steps')
		self.assertContains(dashboard_response, 'Condition estimates')

	def test_dataset_age_bands_match_brfss_labels(self):
		self.assertEqual(_age_category(18), '18-24')
		self.assertEqual(_age_category(27), '25-29')
		self.assertEqual(_age_category(82), '80 or older')

	def test_disease_prediction_returns_percent_and_excludes_target_field(self):
		class ExampleModel:
			classes_ = [0, 1]

			def predict_proba(self, features):
				self.feature_columns = list(features.columns)
				return [[0.72, 0.28]]

		model = ExampleModel()
		bundle = {
			'model_type': 'multi_disease_random_forest',
			'models': {
				'diabetes': {
					'pipeline': model,
					'features': ['BMI', 'PhysicalHealth', 'Sex'],
				},
			},
		}
		with patch('health.risk._load_model', return_value=bundle):
			predictions = predict_disease_risks(self.profile)

		self.assertEqual(predictions[0]['label'], 'Diabetes')
		self.assertEqual(predictions[0]['probability_pct'], 28.0)
		self.assertNotIn('Diabetic', model.feature_columns)

	def test_2022_rows_combine_heart_attack_and_angina_outcomes(self):
		frame = pd.DataFrame({
			'HadHeartAttack': ['No', 'Yes', 'No'],
			'HadAngina': ['Yes', 'No', 'No'],
			'BMI': [24.0, 29.0, 22.0],
			'Sex': ['Female', 'Male', 'Female'],
			'AgeCategory': ['Age 45 to 49', 'Age 65 to 69', 'Age 80 or older'],
			'HadDiabetes': ['No', 'Yes', 'No, pre-diabetes or borderline diabetes'],
			'PhysicalActivities': ['Yes', 'No', 'Yes'],
			'GeneralHealth': ['Good', 'Fair', 'Excellent'],
			'PhysicalHealthDays': [0, 7, 88],
			'MentalHealthDays': [0, 0, 0],
			'DifficultyWalking': ['No', 'Yes', 'No'],
			'SleepHours': [7, 6, 8],
			'HadAsthma': ['No', 'Yes', 'No'],
			'HadKidneyDisease': ['No', 'No', 'No'],
			'HadSkinCancer': ['No', 'No', 'Yes'],
			'SmokerStatus': ['Never smoked', 'Current smoker - now smokes every day', 'Former smoker'],
			'HadStroke': ['No', 'No', 'No'],
		})

		prepared = _prepare_chunk(frame, '2022')

		self.assertEqual(prepared['target_heart_disease'].tolist(), [1.0, 1.0, 0.0])
		self.assertEqual(prepared['target_heart_attack'].tolist(), [0.0, 1.0, 0.0])
		self.assertEqual(prepared['target_angina'].tolist(), [1.0, 0.0, 0.0])
		self.assertEqual(prepared['target_diabetes'].tolist(), [0.0, 1.0, 0.0])
		self.assertEqual(prepared['AgeCategory'].tolist(), ['45-49', '65-69', '80 or older'])
		self.assertEqual(prepared['Diabetic'].tolist(), ['No', 'Yes', 'No, borderline diabetes'])

	def test_raw_brfss_codes_normalize_to_classifier_features(self):
		frame = pd.DataFrame({
			'CVDINFR4': [1, 2], 'CVDCRHD4': [2, 2], 'WEIGHT2': [150, 180],
			'HEIGHT3': [505, 600], 'SMOKE100': [1, 2], 'CVDSTRK3': [2, 2],
			'SEX': [2, 1], '_AGEG5YR': [1, 10], 'DIABETE3': [3, 4],
			'EXERANY2': [1, 2], 'GENHLTH': [3, 4], 'PHYSHLTH': [88, 7],
			'MENTHLTH': [2, 88], 'DIFFWALK': [2, 1], 'SLEPTIM1': [7, 6],
			'ASTHMA3': [2, 1], 'CHCKIDNY': [2, 2], 'CHCSCNCR': [2, 2],
		})

		prepared = _prepare_chunk(frame, 'raw')

		self.assertEqual(prepared['target_heart_disease'].tolist(), [1.0, 0.0])
		self.assertEqual(prepared['target_heart_attack'].tolist(), [1.0, 0.0])
		self.assertEqual(prepared['target_high_blood_pressure'].isna().tolist(), [True, True])
		self.assertEqual(prepared['AgeCategory'].tolist(), ['18-24', '65-69'])
		self.assertEqual(prepared['Sex'].tolist(), ['Female', 'Male'])
		self.assertEqual(prepared['PhysicalHealth'].tolist(), [0.0, 7.0])
		self.assertIn('SkinCancer', prepared.columns)

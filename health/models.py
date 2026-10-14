from django.contrib.auth.models import User
from django.db import models


class HealthProfile(models.Model):
	SEX_CHOICES = [('Female', 'Female'), ('Male', 'Male')]
	DIABETES_CHOICES = [
		('No', 'No'),
		('No, borderline diabetes', 'Borderline'),
		('Yes', 'Yes'),
	]
	HEALTH_CHOICES = [
		('Excellent', 'Excellent'),
		('Very good', 'Very good'),
		('Good', 'Good'),
		('Fair', 'Fair'),
		('Poor', 'Poor'),
	]
	DIET_CHOICES = [('poor', 'Needs work'), ('fair', 'Getting there'), ('good', 'Balanced')]

	user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='health_profile')
	age = models.PositiveSmallIntegerField(default=35)
	sex = models.CharField(max_length=10, choices=SEX_CHOICES, default='Female')
	bmi = models.FloatField(default=24.0)
	smoking = models.BooleanField(default=False)
	alcohol_drinking = models.BooleanField(default=False)
	stroke = models.BooleanField(default=False)
	physical_health_days = models.PositiveSmallIntegerField(default=0)
	mental_health_days = models.PositiveSmallIntegerField(default=0)
	difficulty_walking = models.BooleanField(default=False)
	diabetic = models.CharField(max_length=32, choices=DIABETES_CHOICES, default='No')
	physical_activity = models.BooleanField(default=True)
	general_health = models.CharField(max_length=12, choices=HEALTH_CHOICES, default='Good')
	sleep_hours = models.FloatField(default=7.0)
	asthma = models.BooleanField(default=False)
	kidney_disease = models.BooleanField(default=False)
	skin_cancer = models.BooleanField(default=False)
	diet_quality = models.CharField(max_length=8, choices=DIET_CHOICES, default='fair')
	family_heart_history = models.BooleanField(default=False)
	updated_at = models.DateTimeField(auto_now=True)

	def __str__(self):
		return f'{self.user.username} health profile'


class RiskAssessment(models.Model):
	LEVEL_CHOICES = [('low', 'Low'), ('moderate', 'Moderate'), ('high', 'High')]
	METHOD_CHOICES = [('model', 'Trained ML model'), ('screening', 'Lifestyle screening estimate')]

	profile = models.ForeignKey(HealthProfile, on_delete=models.CASCADE, related_name='assessments')
	risk_score = models.PositiveSmallIntegerField()
	risk_level = models.CharField(max_length=12, choices=LEVEL_CHOICES)
	method = models.CharField(max_length=12, choices=METHOD_CHOICES)
	created_at = models.DateTimeField(auto_now_add=True)

	class Meta:
		ordering = ['-created_at']

	def __str__(self):
		return f'{self.profile.user.username}: {self.risk_level} ({self.risk_score})'

from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.shortcuts import redirect, render
from django.utils import timezone

from .forms import HealthProfileForm
from .models import HealthProfile, RiskAssessment
from .risk import assess_risk, predict_disease_risks, recommendations


def sign_up(request):
	if request.user.is_authenticated:
		return redirect('dashboard')
	form = UserCreationForm(request.POST or None)
	if request.method == 'POST' and form.is_valid():
		user = form.save()
		login(request, user)
		HealthProfile.objects.create(user=user)
		return redirect('assessment')
	return render(request, 'health/signup.html', {'form': form})


def sign_in(request):
	if request.user.is_authenticated:
		return redirect('dashboard')
	form = AuthenticationForm(request, data=request.POST or None)
	if request.method == 'POST' and form.is_valid():
		login(request, form.get_user())
		return redirect('dashboard')
	return render(request, 'health/login.html', {'form': form})


@login_required
def dashboard(request):
	profile, _ = HealthProfile.objects.get_or_create(user=request.user)
	latest = profile.assessments.first()
	history = list(profile.assessments.all()[:6])
	context = {
		'profile': profile,
		'assessment': latest,
		'disease_predictions': predict_disease_risks(profile) if latest else [],
		'history': reversed(history),
		'recommendations': recommendations(profile) if latest else [],
		'today': timezone.localdate(),
	}
	return render(request, 'health/dashboard.html', context)


@login_required
def assessment(request):
	profile, _ = HealthProfile.objects.get_or_create(user=request.user)
	form = HealthProfileForm(request.POST or None, instance=profile)
	if request.method == 'POST' and form.is_valid():
		profile = form.save()
		score, level, method = assess_risk(profile)
		RiskAssessment.objects.create(
			profile=profile,
			risk_score=score,
			risk_level=level,
			method=method,
		)
		messages.success(request, 'Your health profile and assessment have been updated.')
		return redirect('dashboard')
	return render(request, 'health/assessment_form.html', {'form': form})

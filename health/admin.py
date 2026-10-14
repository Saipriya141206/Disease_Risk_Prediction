from django.contrib import admin

from .models import HealthProfile, RiskAssessment


@admin.register(HealthProfile)
class HealthProfileAdmin(admin.ModelAdmin):
	list_display = ('user', 'age', 'general_health', 'updated_at')
	search_fields = ('user__username', 'user__email')


@admin.register(RiskAssessment)
class RiskAssessmentAdmin(admin.ModelAdmin):
	list_display = ('profile', 'risk_score', 'risk_level', 'method', 'created_at')
	list_filter = ('risk_level', 'method', 'created_at')

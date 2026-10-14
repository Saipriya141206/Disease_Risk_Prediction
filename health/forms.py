from django import forms

from .models import HealthProfile


class HealthProfileForm(forms.ModelForm):
    class Meta:
        model = HealthProfile
        exclude = ['user', 'updated_at']
        widgets = {
            'age': forms.NumberInput(attrs={'min': 18, 'max': 100}),
            'bmi': forms.NumberInput(attrs={'min': 10, 'max': 80, 'step': '0.1'}),
            'physical_health_days': forms.NumberInput(attrs={'min': 0, 'max': 30}),
            'mental_health_days': forms.NumberInput(attrs={'min': 0, 'max': 30}),
            'sleep_hours': forms.NumberInput(attrs={'min': 0, 'max': 24, 'step': '0.5'}),
        }

    def clean_age(self):
        age = self.cleaned_data['age']
        if not 18 <= age <= 100:
            raise forms.ValidationError('Enter an age between 18 and 100.')
        return age

    def clean_bmi(self):
        bmi = self.cleaned_data['bmi']
        if not 10 <= bmi <= 80:
            raise forms.ValidationError('Enter a BMI between 10 and 80.')
        return bmi

    def clean_sleep_hours(self):
        hours = self.cleaned_data['sleep_hours']
        if not 0 <= hours <= 24:
            raise forms.ValidationError('Enter a value between 0 and 24 hours.')
        return hours
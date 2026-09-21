from django import forms


class CropRecommendationForm(forms.Form):
    # Soil Nutrients
    N = forms.IntegerField(
        label="Nitrogen (N)",
        min_value=0,
        max_value=140,
        initial=90,
        help_text="Nitrogen content in soil (0 – 140 kg/ha)",
        widget=forms.NumberInput(
            attrs={
                "class": "form-input",
                "placeholder": "e.g. 90",
                "min": 0,
                "max": 140,
            }
        ),
    )
    P = forms.IntegerField(
        label="Phosphorus (P)",
        min_value=5,
        max_value=145,
        initial=42,
        help_text="Phosphorus content in soil (5 – 145 kg/ha)",
        widget=forms.NumberInput(
            attrs={
                "class": "form-input",
                "placeholder": "e.g. 42",
                "min": 5,
                "max": 145,
            }
        ),
    )
    K = forms.IntegerField(
        label="Potassium (K)",
        min_value=5,
        max_value=205,
        initial=43,
        help_text="Potassium content in soil (5 – 205 kg/ha)",
        widget=forms.NumberInput(
            attrs={
                "class": "form-input",
                "placeholder": "e.g. 43",
                "min": 5,
                "max": 205,
            }
        ),
    )
    ph = forms.FloatField(
        label="Soil pH",
        min_value=3.5,
        max_value=10.0,
        initial=6.5,
        help_text="Soil pH value (3.5 – 10.0)",
        widget=forms.NumberInput(
            attrs={
                "class": "form-input",
                "placeholder": "e.g. 6.5",
                "min": 3.5,
                "max": 10.0,
                "step": "0.1",
            }
        ),
    )

    # Climate Conditions
    temperature = forms.FloatField(
        label="Temperature",
        min_value=8.0,
        max_value=45.0,
        initial=20.9,
        help_text="Ambient temperature in Celsius (8.0 – 45.0 °C)",
        widget=forms.NumberInput(
            attrs={
                "class": "form-input",
                "placeholder": "e.g. 20.9",
                "min": 8.0,
                "max": 45.0,
                "step": "0.1",
            }
        ),
    )
    humidity = forms.FloatField(
        label="Relative Humidity",
        min_value=14.0,
        max_value=100.0,
        initial=82.0,
        help_text="Relative humidity percentage (14.0 – 100.0 %)",
        widget=forms.NumberInput(
            attrs={
                "class": "form-input",
                "placeholder": "e.g. 82.0",
                "min": 14.0,
                "max": 100.0,
                "step": "0.1",
            }
        ),
    )
    rainfall = forms.FloatField(
        label="Rainfall",
        min_value=20.0,
        max_value=300.0,
        initial=202.9,
        help_text="Annual/seasonal rainfall in mm (20.0 – 300.0 mm)",
        widget=forms.NumberInput(
            attrs={
                "class": "form-input",
                "placeholder": "e.g. 202.9",
                "min": 20.0,
                "max": 300.0,
                "step": "0.1",
            }
        ),
    )

    # Convenience helper property to iterate fields by section in templates
    @property
    def soil_fields(self):
        return [self["N"], self["P"], self["K"], self["ph"]]

    @property
    def climate_fields(self):
        return [self["temperature"], self["humidity"], self["rainfall"]]

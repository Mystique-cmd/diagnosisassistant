# ============================================================
# MODULE 6: Fuzzy Logic — Patient Severity Assessment
# Covers: Week 12 (Fuzzy Logic)
# ============================================================

from typing import Dict, Any, Union


class FuzzySeverityAssessor:
    """
    Fuzzy logic system for patient severity assessment.
    Inputs:  Temperature, Heart Rate, Symptom Count
    Output:  Severity Score (0-100)
    """

    def _tri(self, x: float, a: float, b: float, c: float) -> float:
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return max(0.0, (x - a) / (b - a))
        return max(0.0, (c - x) / (c - b))

    def _trap(self, x: float, a: float, b: float, c: float, d: float) -> float:
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if a < x < b:
            return (x - a) / (b - a)
        return (d - x) / (d - c)

    def _membership_temp(self, temp: float) -> Dict[str, float]:
        """Temperature membership functions (triangular / trapezoidal)"""
        return {
            'normal':   self._trap(temp, 30.0, 35.0, 37.0, 37.8),
            'mild':     self._tri(temp, 37.0, 38.0, 39.0),
            'high':     self._tri(temp, 38.0, 39.0, 40.5),
            'critical': self._trap(temp, 39.5, 40.5, 45.0, 50.0)
        }

    def _membership_hr(self, hr: float) -> Dict[str, float]:
        """Heart rate membership functions (triangular / trapezoidal)"""
        return {
            'low':      self._trap(hr, 20, 30, 55, 70),
            'normal':   self._tri(hr, 60, 75, 95),
            'elevated': self._tri(hr, 85, 100, 115),
            'high':     self._trap(hr, 105, 120, 180, 250)
        }

    def _membership_symptoms(self, count: float) -> Dict[str, float]:
        """Symptom count membership functions (tri / trap)"""
        return {
            'few':      self._trap(count, 0, 0, 1, 3),
            'moderate': self._tri(count, 2, 4, 6),
            'many':     self._trap(count, 5, 7, 20, 50)
        }

    def _defuzzify(self, severity_rules: Dict[str, float]) -> float:
        """Centroid defuzzification with safe fallback"""
        centers = {
            'low': 15.0,
            'mild': 35.0,
            'moderate': 55.0,
            'high': 75.0,
            'critical': 92.0
        }
        numerator = 0.0
        denominator = 0.0
        for k, v in severity_rules.items():
            if k in centers:
                numerator += centers[k] * v
                denominator += v
        if denominator == 0.0:
            # Fallback to a neutral baseline score if rule coverage is 0
            return 30.0
        return numerator / denominator

    def assess(self, temperature: float, heart_rate: int,
               symptom_count: int) -> Dict:
        """Full fuzzy inference pipeline"""
        temp_mf    = self._membership_temp(temperature)
        hr_mf      = self._membership_hr(heart_rate)
        symptom_mf = self._membership_symptoms(symptom_count)

        rules = {
            'critical': max(
                min(temp_mf['critical'], hr_mf['high']),
                min(temp_mf['critical'], symptom_mf['many'])
            ),
            'high': max(
                min(temp_mf['high'], hr_mf['elevated']),
                min(temp_mf['high'], symptom_mf['many']),
                min(temp_mf['mild'], hr_mf['high'])
            ),
            'moderate': max(
                min(temp_mf['mild'], hr_mf['normal']),
                min(temp_mf['high'], symptom_mf['moderate']),
                min(temp_mf['normal'], symptom_mf['many']),
                min(hr_mf['elevated'], symptom_mf['moderate'])
            ),
            'mild': max(
                min(temp_mf['mild'], symptom_mf['few']),
                min(temp_mf['normal'], symptom_mf['moderate']),
                min(temp_mf['mild'], symptom_mf['moderate'])
            ),
            'low': min(temp_mf['normal'], hr_mf['normal'],
                       symptom_mf['few'])
        }

        severity_score = self._defuzzify(rules)
        severity_label = self._classify(severity_score)

        return {
            'severity_score': round(severity_score, 2),
            'severity_label': severity_label,
            'rule_strengths': {k: round(v, 3) for k, v in rules.items()},
            'memberships': {
                'temperature': temp_mf,
                'heart_rate':   hr_mf,
                'symptoms':     symptom_mf
            }
        }

    def _classify(self, score: float) -> str:
        if score >= 80:
            return "CRITICAL"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MODERATE"
        elif score >= 20:
            return "MILD"
        return "LOW"

    def _extract_symptom_count(self, raw_symptoms: Any) -> int:
        """Robust helper to extract numeric symptom count"""
        if isinstance(raw_symptoms, (list, tuple, set)):
            return len(raw_symptoms)
        elif isinstance(raw_symptoms, (int, float)):
            return int(raw_symptoms)
        elif isinstance(raw_symptoms, str) and raw_symptoms.isdigit():
            return int(raw_symptoms)
        return 0

    def analyze(self, percept: Union[Dict, Any]) -> Dict:
        """Module interface for the agent (handles dict and PatientPercept objects)"""
        if isinstance(percept, dict):
            temp = float(percept.get('temperature', percept.get('temp', 37.0)))
            hr = int(percept.get('heart_rate', percept.get('hr', 70)))
            raw_symptoms = percept.get('symptoms', percept.get('symptom_count', []))
        else:
            temp = float(getattr(percept, 'temperature', 37.0))
            hr = int(getattr(percept, 'heart_rate', 70))
            raw_symptoms = getattr(percept, 'symptoms', getattr(percept, 'symptom_count', []))

        symptom_count = self._extract_symptom_count(raw_symptoms)

        result = self.assess(temp, hr, symptom_count)
        result['module']    = 'Fuzzy Assessor'
        result['summary']   = f"Severity: {result['severity_label']} ({result['severity_score']:.1f}/100)"
        result['diagnosis'] = result['severity_label']
        result['confidence']= result['severity_score'] / 100.0
        return result


if __name__ == "__main__":
    assessor = FuzzySeverityAssessor()
    
    # Test case 1: Dictionary with list of symptoms
    dict_percept = {'temp': 38.8, 'hr': 102, 'symptoms': ['fever', 'cough', 'fatigue']}
    print("Dict Test:", assessor.analyze(dict_percept)['summary'])
    
    # Test case 2: Numeric symptom count directly passed
    num_percept = {'temperature': 39.2, 'heart_rate': 110, 'symptoms': 5}
    print("Numeric Symptom Test:", assessor.analyze(num_percept)['summary'])
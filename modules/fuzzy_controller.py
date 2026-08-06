# ============================================================
# MODULE 6: Fuzzy Logic — Patient Severity Assessment
# Covers: Week 12 (Fuzzy Logic)
# ============================================================

from typing import Dict

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
            'normal':   self._trap(temp, 34.0, 36.5, 37.0, 37.8),
            'mild':     self._tri(temp, 37.0, 38.0, 39.0),
            'high':     self._tri(temp, 38.5, 39.5, 40.5),
            'critical': self._trap(temp, 39.0, 40.0, 45.0, 50.0)
        }

    def _membership_hr(self, hr: int) -> Dict[str, float]:
        """Heart rate membership functions (triangular / trapezoidal)"""
        return {
            'low':      self._tri(hr, 30, 55, 70),
            'normal':   self._tri(hr, 65, 80, 95),
            'elevated': self._tri(hr, 90, 100, 110),
            'high':     self._trap(hr, 100, 115, 160, 200)
        }

    def _membership_symptoms(self, count: int) -> Dict[str, float]:
        """Symptom count membership functions (tri / trap)"""
        return {
            'few':      self._tri(count, 0, 1, 3),
            'moderate': self._tri(count, 2, 4, 6),
            'many':     self._trap(count, 5, 7, 20, 30)
        }

    def _defuzzify(self, severity_rules: Dict[str, float]) -> float:
        """Centroid defuzzification"""
        centers = {'low': 15, 'mild': 35, 'moderate': 55,
                   'high': 75, 'critical': 92}
        numerator = 0.0
        denominator = 0.0
        for k, v in severity_rules.items():
            if k in centers:
                numerator += centers[k] * v
                denominator += v
        if denominator == 0.0:
            return 0.0
        return numerator / denominator

    def assess(self, temperature: float, heart_rate: int,
               symptom_count: int) -> Dict:
        """Full fuzzy inference pipeline"""
        # Fuzzification
        temp_mf    = self._membership_temp(temperature)
        hr_mf      = self._membership_hr(heart_rate)
        symptom_mf = self._membership_symptoms(symptom_count)

        # Rule evaluation (min for AND, max for OR)
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

        # Defuzzification
        severity_score = self._defuzzify(rules)
        severity_label = self._classify(severity_score)

        return {
            'severity_score': round(severity_score, 2),
            'severity_label': severity_label,
            'rule_strengths': {k: round(v, 3) for k, v in rules.items()},
            'memberships': {
                'temperature': temp_mf,
                'heart_rate':  hr_mf,
                'symptoms':    symptom_mf
            }
        }

    def _classify(self, score: float) -> str:
        if score >= 80: return "CRITICAL"
        elif score >= 60: return "HIGH"
        elif score >= 40: return "MODERATE"
        elif score >= 20: return "MILD"
        return "LOW"

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        result = self.assess(
            percept.temperature,
            percept.heart_rate,
            len(percept.symptoms)
        )
        result['summary']   = (f"Severity: {result['severity_label']} "
                               f"({result['severity_score']:.1f}/100)")
        result['diagnosis'] = result['severity_label']
        result['confidence']= result['severity_score'] / 100
        return result

if __name__ == "__main__":
    fa = FuzzySeverityAssessor()

    print("Running Module 6: Fuzzy Logic Tests...\n")

    # These are the exact test cases required by the Capstone manual
    test_cases = [
        (37.0, 72, 2, "Normal patient"),
        (38.5, 95, 4, "Mild illness"),
        (39.8, 115, 7, "Severe case"),
        (40.2, 130, 9, "Critical case"),
    ]

    for temp, hr, count, desc in test_cases:
        result = fa.assess(temp, hr, count)
        # Formatting exactly as the lab manual dictates
        print(f"{desc:15}: Score={result['severity_score']:>4}, Label={result['severity_label']}")

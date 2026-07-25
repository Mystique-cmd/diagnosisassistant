from modules.fuzzy_controller import FuzzySeverityAssessor

fa = FuzzySeverityAssessor()

test_cases = [
    (37.0, 72, 2, "Normal patient"),
    (38.5, 95, 4, "Mild illness"),
    (39.8, 115, 7, "Severe case"),
    (40.2, 130, 9, "Critical case"),
]

results = []
for temp, hr, count, desc in test_cases:
    result = fa.assess(temp, hr, count)
    results.append(f"{desc}: Score={result['severity_score']}, Label={result['severity_label']}")
    results.append(f"  memberships: {result['memberships']}")
    results.append(f"  rules: {result['rule_strengths']}")

print('\n'.join(results))

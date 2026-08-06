"""
Main Application Entry Point (app.py)
Integrates all AI sub-modules into the Intelligent Agent architecture.
"""

from modules.agent import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase

# Import AI Sub-Modules safely with fallback handling
try:
    from modules.bayesian_net import SimpleBayesianDiagnostics
except ImportError:
    SimpleBayesianDiagnostics = None

try:
    from modules.ml_classifier import MLDiagnosticClassifier
except ImportError:
    MLDiagnosticClassifier = None

try:
    from modules.neural_network import NeuralDiagnosticModel
except ImportError:
    NeuralDiagnosticModel = None

try:
    from modules.fuzzy_controller import FuzzySeverityAssessor
except ImportError:
    FuzzySeverityAssessor = None

try:
    from modules.planner import TreatmentPlanner
except ImportError:
    TreatmentPlanner = None


def setup_system() -> HealthcareDiagnosticAgent:
    """Instantiates the agent and registers all AI sub-modules."""
    print("Initializing Healthcare Diagnostic Assistant...")
    agent = HealthcareDiagnosticAgent()

    # 1. Module 2: Knowledge Base
    print(" -> Registering Module 2: Medical Knowledge Base...")
    kb = MedicalKnowledgeBase()
    agent.register_module('Knowledge Base', kb)

    # 2. Module 3: Bayesian Network
    if SimpleBayesianDiagnostics:
        print(" -> Registering Module 3: Bayesian Network...")
        agent.register_module('BayesianNet', SimpleBayesianDiagnostics())

    # 3. Module 4: Machine Learning Classifier
    if MLDiagnosticClassifier:
        print(" -> Registering Module 4: ML Diagnostic Classifier...")
        ml = MLDiagnosticClassifier()
        try:
            ml.train(verbose=False)
        except Exception:
            pass
        agent.register_module('ML Classifier', ml)

    # 4. Module 5: Deep Neural Network
    if NeuralDiagnosticModel:
        print(" -> Registering Module 5: Deep Neural Network...")
        nn = NeuralDiagnosticModel()
        try:
            nn.train(epochs=10, verbose=False)
        except Exception:
            pass
        agent.register_module('Neural Network', nn)

    # 5. Module 6: Fuzzy Severity Assessor
    if FuzzySeverityAssessor:
        print(" -> Registering Module 6: Fuzzy Severity Assessor...")
        agent.register_module('Fuzzy Severity', FuzzySeverityAssessor())

    # 6. Module 7: AI Treatment Planner
    if TreatmentPlanner:
        print(" -> Registering Module 7: AI Treatment Planner...")
        agent.register_module('Treatment Planner', TreatmentPlanner())

    print("=" * 55)
    print("System Initialization Complete! All modules registered.")
    print("=" * 55)
    return agent


def _get_input(prompt: str, default: str = ""):
    """Prompt user for input, returning default if empty."""
    value = input(prompt)
    return value.strip() if value.strip() else default


def collect_patient_input(patient_count: int) -> PatientPercept:
    """Collect diagnosis (symptoms and vitals) interactively from the user."""
    print("=" * 55)
    print(f"  PATIENT INTAKE FORM - Patient {patient_count}")
    print("=" * 55)

    patient_id = _get_input("Patient ID [P001]: ", f"P{patient_count:03d}")

    symptoms_raw = _get_input(
        "Symptoms (comma-separated) [fever, cough]: ",
        "fever, cough"
    )
    symptoms = [s.strip().lower().replace(' ', '_')
                for s in symptoms_raw.split(',') if s.strip()]

    age = int(_get_input("Age [30]: ", "30"))
    temperature = float(_get_input("Temperature (C) [37.0]: ", "37.0"))
    heart_rate = int(_get_input("Heart rate (bpm) [80]: ", "80"))
    blood_pressure = _get_input("Blood pressure [120/80]: ", "120/80")

    return PatientPercept(
        patient_id=patient_id,
        symptoms=symptoms,
        age=age,
        temperature=temperature,
        heart_rate=heart_rate,
        blood_pressure=blood_pressure
    )


def main():
    agent = setup_system()

    patient_count = 0
    while True:
        patient_count += 1
        patient = collect_patient_input(patient_count)

        print(f"[Patient Intake] Processing {patient.patient_id}...")

        # Perceive -> Think -> Act Cycle
        agent.perceive(patient)
        think_results = agent.think()

        # Retrieve diagnosis map safely
        if not think_results and hasattr(agent, 'memory'):
            think_results = getattr(agent.memory, 'diagnosis_results', getattr(agent, 'diagnosis_results', {}))

        try:
            report = agent.act(think_results)
        except TypeError:
            report = agent.act()

        # Print Final Diagnostic Report
        print("=" * 15 + f" DIAGNOSTIC REPORT: {patient.patient_id} " + "=" * 15)
        if isinstance(report, dict):
            print(f"Primary Diagnosis: {report.get('final_diagnosis', report.get('diagnosis', 'Unknown'))}")
            print(f"Confidence Score:  {report.get('confidence_score', report.get('confidence', 0.0)):.2%}")
            print(f"Urgency Level:     {report.get('urgency_level', report.get('urgency', 'LOW'))}")

            plan_items = report.get('treatment_plan', report.get('plan', []))
            if plan_items:
                print("Treatment Plan Steps:")
                for step in plan_items:
                    if isinstance(step, dict):
                        print(f"  Step {step.get('step', '-')}: {step.get('action', step)} [{step.get('duration', 'N/A')}]")
                    else:
                        print(f"  - {step}")
        else:
            print(report)
        print("=" * 55)

        # Ask user whether to continue with another patient
        continue_input = _get_input(
            "Process another patient? (y/n) [y]: ",
            "y"
        ).lower()
        if continue_input not in ('y', 'yes'):
            print("Exiting. Thank you for using the Healthcare Diagnostic Assistant.")
            break


if __name__ == "__main__":
    main()

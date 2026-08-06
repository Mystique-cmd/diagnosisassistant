"""
Main Application Entry Point (app.py)
Integrates all AI sub-modules into the Intelligent Agent architecture.

This version provides a Graphical User Interface (Tkinter) for easier
patient onboarding and use of the system. Instead of typing symptom
names manually, users can click symptom checkboxes and fill in vitals.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

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


# ------------------------------------------------------------------
# Known symptom vocabulary (human-readable label -> snake_case key)
# Derived from data/final_training_data.csv and modules/bayesian_net.py
# ------------------------------------------------------------------
SYMPTOM_OPTIONS = [
    ("Fever",                    "fever"),
    ("Cough",                    "cough"),
    ("Fatigue",                  "fatigue"),
    ("Difficulty Breathing",     "difficulty_breathing"),
    ("Muscle Ache",              "muscle_ache"),
    ("Body Aches",               "body_aches"),
    ("Headache",                 "headache"),
    ("Sore Throat",              "sore_throat"),
    ("Nausea",                   "nausea"),
    ("Chills",                   "chills"),
    ("Vomiting",                 "vomiting"),
    ("Diarrhea",                 "diarrhea"),
    ("Rash",                     "rash"),
    ("Loss of Smell",            "loss_of_smell"),
    ("Chest Pain",               "chest_pain"),
    ("Shortness of Breath",      "shortness_of_breath"),
    ("Joint Pain",               "joint_pain"),
    ("Sweating",                 "sweating"),
    ("Abdominal Pain",           "abdominal_pain"),
    ("Runny Nose",               "runny_nose"),
    ("Frequent Urination",       "frequent_urination"),
    ("Excessive Thirst",         "excessive_thirst"),
    ("Blurred Vision",           "blurred_vision"),
]

# Urgency -> display color mapping
URGENCY_COLORS = {
    "CRITICAL": "#d32f2f",
    "HIGH":     "#f57c00",
    "MEDIUM":   "#fbc02d",
    "LOW":      "#388e3c",
}


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


class DiagnosticGUI:
    """Tkinter GUI for patient onboarding and displaying diagnosis results."""

    def __init__(self, root: tk.Tk, agent: HealthcareDiagnosticAgent):
        self.root = root
        self.agent = agent
        self.patient_count = 0

        root.title("Healthcare Diagnostic Assistant")
        root.geometry("980x760")
        root.minsize(860, 640)

        # Holds the symptom checkbox variables
        self.symptom_vars = {}

        self._build_layout()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------
    def _build_layout(self):
        # Main container with two columns: input (left) and results (right)
        main = ttk.Frame(self.root, padding=(12, 12, 12, 12))
        main.pack(fill=tk.BOTH, expand=True)

        # Left: patient intake form
        left = ttk.LabelFrame(main, text=" Patient Intake Form ", padding=(12, 12))
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 12))

        self._build_patient_info(left)
        self._build_symptom_panel(left)
        self._build_action_buttons(left)

        # Right: results panel
        right = ttk.LabelFrame(main, text=" Diagnostic Report ", padding=(12, 12))
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_results_panel(right)

    def _build_patient_info(self, parent):
        info = ttk.Frame(parent)
        info.pack(fill=tk.X, pady=(0, 10))

        # Patient ID
        ttk.Label(info, text="Patient ID:").grid(row=0, column=0, sticky="w", pady=2)
        self.patient_id_var = tk.StringVar(value="P001")
        ttk.Entry(info, textvariable=self.patient_id_var, width=18).grid(
            row=0, column=1, sticky="w", pady=2, padx=(6, 0))

        # Age
        ttk.Label(info, text="Age:").grid(row=1, column=0, sticky="w", pady=2)
        self.age_var = tk.StringVar(value="30")
        ttk.Entry(info, textvariable=self.age_var, width=18).grid(
            row=1, column=1, sticky="w", pady=2, padx=(6, 0))

        # Temperature
        ttk.Label(info, text="Temperature (°C):").grid(row=2, column=0, sticky="w", pady=2)
        self.temperature_var = tk.StringVar(value="37.0")
        ttk.Entry(info, textvariable=self.temperature_var, width=18).grid(
            row=2, column=1, sticky="w", pady=2, padx=(6, 0))

        # Heart rate
        ttk.Label(info, text="Heart Rate (bpm):").grid(row=3, column=0, sticky="w", pady=2)
        self.heart_rate_var = tk.StringVar(value="80")
        ttk.Entry(info, textvariable=self.heart_rate_var, width=18).grid(
            row=3, column=1, sticky="w", pady=2, padx=(6, 0))

        # Blood pressure
        ttk.Label(info, text="Blood Pressure:").grid(row=4, column=0, sticky="w", pady=2)
        self.blood_pressure_var = tk.StringVar(value="120/80")
        ttk.Entry(info, textvariable=self.blood_pressure_var, width=18).grid(
            row=4, column=1, sticky="w", pady=2, padx=(6, 0))

    def _build_symptom_panel(self, parent):
        sym_frame = ttk.LabelFrame(parent, text=" Select Symptoms ", padding=(8, 8))
        sym_frame.pack(fill=tk.X, pady=(0, 10))

        # Scrollable symptom checklist
        canvas = tk.Canvas(sym_frame, height=260, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sym_frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)

        scrollable.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Place checkboxes in a grid (2 columns)
        for index, (label, key) in enumerate(SYMPTOM_OPTIONS):
            var = tk.BooleanVar(value=False)
            self.symptom_vars[key] = var
            cb = ttk.Checkbutton(
                scrollable, text=label, variable=var,
                command=self._on_symptom_toggle
            )
            row, col = divmod(index, 2)
            cb.grid(row=row, column=col, sticky="w", padx=4, pady=2)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _build_action_buttons(self, parent):
        btns = ttk.Frame(parent)
        btns.pack(fill=tk.X, pady=(4, 0))

        self.run_button = ttk.Button(
            btns, text="▶ Run Diagnosis",
            command=self.run_diagnosis
        )
        self.run_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 6))

        clear_btn = ttk.Button(
            btns, text="Clear",
            command=self.clear_form
        )
        clear_btn.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(6, 0))

    def _build_results_panel(self, parent):
        # Header with urgency badge
        header = ttk.Frame(parent)
        header.pack(fill=tk.X, pady=(0, 8))

        self.urgency_label = ttk.Label(
            header, text="No diagnosis yet",
            font=("Helvetica", 11, "bold"),
            foreground="#333333"
        )
        self.urgency_label.pack(side=tk.LEFT)

        self.confidence_label = ttk.Label(
            header,
            text="",
            font=("Helvetica", 10)
        )
        self.confidence_label.pack(side=tk.RIGHT)

        # Scrollable report text
        self.report_text = scrolledtext.ScrolledText(
            parent, wrap=tk.WORD, height=20,
            font=("Consolas", 10)
        )
        self.report_text.pack(fill=tk.BOTH, expand=True)
        self.report_text.configure(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_symptom_toggle(self):
        """Update selected-symptom count in title (optional)."""
        selected = self._get_selected_symptoms()
        count = len(selected)
        if count:
            self.root.title(
                f"Healthcare Diagnostic Assistant — {count} symptom(s) selected"
            )
        else:
            self.root.title("Healthcare Diagnostic Assistant")

    def _get_selected_symptoms(self) -> list:
        """Return the list of snake_case symptom keys currently checked."""
        return [
            key for key, var in self.symptom_vars.items()
            if var.get()
        ]

    def _collect_percept(self) -> PatientPercept:
        """Read form fields and build a PatientPercept object."""
        self.patient_count += 1

        patient_id = self.patient_id_var.get().strip() or f"P{self.patient_count:03d}"
        symptoms = self._get_selected_symptoms()

        # Validate numeric fields
        try:
            age = int(self.age_var.get().strip() or "30")
        except ValueError:
            raise ValueError("Age must be a whole number.")
        try:
            temperature = float(self.temperature_var.get().strip() or "37.0")
        except ValueError:
            raise ValueError("Temperature must be a number.")
        try:
            heart_rate = int(self.heart_rate_var.get().strip() or "80")
        except ValueError:
            raise ValueError("Heart rate must be a whole number.")

        blood_pressure = self.blood_pressure_var.get().strip() or "120/80"

        return PatientPercept(
            patient_id=patient_id,
            symptoms=symptoms,
            age=age,
            temperature=temperature,
            heart_rate=heart_rate,
            blood_pressure=blood_pressure
        )

    def run_diagnosis(self):
        """Collect input, run the agent cycle, and display the report."""
        try:
            patient = self._collect_percept()
        except ValueError as exc:
            messagebox.showerror("Invalid Input", str(exc))
            return

        if not patient.symptoms:
            messagebox.showwarning(
                "No Symptoms Selected",
                "Please select at least one symptom before running the diagnosis."
            )
            self.patient_count -= 1
            return

        # Perceive -> Think -> Act Cycle
        self.agent.perceive(patient)
        think_results = self.agent.think()

        # Retrieve diagnosis map safely
        if not think_results and hasattr(self.agent, 'memory'):
            think_results = getattr(
                self.agent.memory,
                'diagnosis_results',
                getattr(self.agent, 'diagnosis_results', {})
            )

        try:
            report = self.agent.act(think_results)
        except TypeError:
            report = self.agent.act()

        self._display_report(report, patient)

    def _display_report(self, report, patient):
        """Render the diagnostic report in the results panel."""
        if isinstance(report, dict):
            diagnosis = report.get(
                'diagnosis',
                report.get('final_diagnosis', 'Unknown')
            )
            confidence = report.get('confidence', report.get('confidence_score', 0.0))
            urgency = report.get(
                'urgency',
                report.get('urgency_level', 'LOW')
            )

            # Update header
            color = URGENCY_COLORS.get(str(urgency).upper(), "#333333")
            self.urgency_label.configure(
                text=f"Patient {patient.patient_id} — Urgency: {urgency}",
                foreground=color
            )
            self.confidence_label.configure(
                text=f"Confidence: {float(confidence):.2%}"
            )

            # Build report text
            lines = []
            lines.append(f"DIAGNOSTIC REPORT: {patient.patient_id}")
            lines.append("=" * 55)
            lines.append(f"Primary Diagnosis : {diagnosis}")
            lines.append(f"Confidence Score  : {float(confidence):.2%}")
            lines.append(f"Urgency Level     : {urgency}")
            lines.append(f"Next Action       : "
                         f"{report.get('next_action', 'N/A')}")
            lines.append("")
            lines.append(f"Reported Symptoms : {', '.join(patient.symptoms) or 'None'}")
            lines.append("")

            # Treatment plan
            plan_items = report.get('treatment_plan', report.get('plan', []))
            if plan_items:
                lines.append("Treatment Plan Steps:")
                for step in plan_items:
                    if isinstance(step, dict):
                        duration = step.get('duration', 'N/A')
                        lines.append(
                            f"  Step {step.get('step', '-')}: "
                            f"{step.get('action', step)} [{duration}]"
                        )
                    else:
                        lines.append(f"  - {step}")
                lines.append("")

            # Recommendations
            recommendations = report.get('recommendations', [])
            if recommendations:
                lines.append("Recommendations:")
                for rec in recommendations:
                    lines.append(f"  • {rec}")

            self._set_report_text("\n".join(lines))
        else:
            self._set_report_text(str(report))

    def _set_report_text(self, content: str):
        """Replace the report panel content."""
        self.report_text.configure(state=tk.NORMAL)
        self.report_text.delete("1.0", tk.END)
        self.report_text.insert(tk.END, content)
        self.report_text.configure(state=tk.DISABLED)

    def clear_form(self):
        """Reset all form fields and clear the report."""
        self.patient_id_var.set(f"P{self.patient_count + 1:03d}")
        self.age_var.set("30")
        self.temperature_var.set("37.0")
        self.heart_rate_var.set("80")
        self.blood_pressure_var.set("120/80")

        for var in self.symptom_vars.values():
            var.set(False)

        self.urgency_label.configure(
            text="No diagnosis yet",
            foreground="#333333"
        )
        self.confidence_label.configure(text="")
        self._set_report_text("")
        self.root.title("Healthcare Diagnostic Assistant")


def main():
    agent = setup_system()

    root = tk.Tk()
    app = DiagnosticGUI(root, agent)
    root.mainloop()

    print("\nExiting. Thank you for using the Healthcare Diagnostic Assistant.")


if __name__ == "__main__":
    main()

# ============================================================
# MODULE 8: Automated Planning — STRIPS Treatment Planner
# Covers: Week 14 (Automated Planning)
# ============================================================

from collections import deque
from typing import Dict, List, Any, Optional, Set, Union


class TreatmentPlanner:
    """
    STRIPS-based automated planner for generating patient treatment plans.

    The planner performs forward state-space search (Breadth-First Search)
    over a library of STRIPS actions to reach a goal state from an initial
    state derived from the diagnosis.

    Uniform STRIPS action schema:
        name      : str   — action identifier
        preconds  : Set   — predicates that must hold before the action
        add       : Set   — predicates added to the state by the action
        del       : Set   — predicates removed from the state by the action
        duration  : str   — human-readable action duration
        cost      : int   — action cost (used for potential cost-aware search)
    """

    def __init__(self):
        # Action Domain Definition (STRIPS Operators)
        self.action_library = [
            {
                'name': 'IsolatePatient',
                'preconds': {'CONTAGIOUS_DISEASE'},
                'add': {'PATIENT_ISOLATED'},
                'del': set(),
                'duration': '14 days',
                'cost': 1,
            },
            {
                'name': 'OrderPCRTest',
                'preconds': {'COVID_SUSPECTED'},
                'add': {'PCR_ORDERED'},
                'del': set(),
                'duration': '24 hours',
                'cost': 1,
            },
            {
                'name': 'ReceivePCRResult',
                'preconds': {'PCR_ORDERED'},
                'add': {'PCR_CONFIRMED'},
                'del': {'COVID_SUSPECTED'},
                'duration': '24 hours',
                'cost': 1,
            },
            {
                'name': 'PrescribeAntiviral',
                'preconds': {'VIRAL_INFECTION'},
                'add': {'TREATMENT_STARTED'},
                'del': set(),
                'duration': '10 minutes',
                'cost': 1,
            },
            {
                'name': 'AdministerOxygen',
                'preconds': {'RESPIRATORY_DISTRESS'},
                'add': {'OXYGEN_ADMINISTERED'},
                'del': set(),
                'duration': 'Immediate',
                'cost': 1,
            },
            {
                'name': 'PrescribeAntibiotics',
                'preconds': {'BACTERIAL_INFECTION'},
                'add': {'TREATMENT_STARTED'},
                'del': set(),
                'duration': '10 minutes',
                'cost': 1,
            },
            {
                'name': 'AdministerAspirin',
                'preconds': {'CARDIAC_SUSPECTED'},
                'add': {'CARDIAC_INITIAL_CARE', 'TREATMENT_STARTED'},
                'del': set(),
                'duration': 'Immediate',
                'cost': 1,
            },
            {
                'name': 'AdmitToICU',
                'preconds': {'EMERGENCY_CASE', 'ICU_AVAILABLE'},
                'add': {'PATIENT_IN_ICU'},
                'del': set(),
                'duration': 'Immediate',
                'cost': 1,
            },
            {
                'name': 'MonitorVitals',
                'preconds': {'PATIENT_PRESENT'},
                'add': {'VITALS_MONITORED'},
                'del': set(),
                'duration': 'Continuous',
                'cost': 1,
            },
            {
                'name': 'ScheduleFollowUp',
                'preconds': {'TREATMENT_STARTED'},
                'add': {'FOLLOWUP_SCHEDULED'},
                'del': set(),
                'duration': '5 minutes',
                'cost': 1,
            },
            {
                'name': 'RestAndHydrate',
                'preconds': {'MILD_ILLNESS'},
                'add': {'TREATMENT_STARTED'},
                'del': set(),
                'duration': '3-5 days',
                'cost': 1,
            },
            {
                'name': 'DischargePatient',
                'preconds': {'PLAN_COMPLETE', 'SYMPTOMS_RESOLVED'},
                'add': {'PATIENT_DISCHARGED'},
                'del': {'PLAN_COMPLETE'},
                'duration': '30 minutes',
                'cost': 0,
            },
        ]

        # Backwards-compatible alias: action name -> action dict
        self.actions = {a['name']: a for a in self.action_library}

    # ------------------------------------------------------------------
    # STRIPS state-transition helpers
    # ------------------------------------------------------------------
    def _apply_action(self, state: frozenset,
                      action: Dict) -> Optional[frozenset]:
        """Return the successor state if the action is applicable, else None."""
        if not action['preconds'].issubset(state):
            return None
        return frozenset((state - action['del']) | action['add'])

    def generate_plan(self,
                      initial_state: Set[str],
                      goal_state:    Set[str]) -> Optional[List[Dict]]:
        """
        BFS-based forward state-space plan generation.

        Returns the minimal-length sequence of STRIPS actions that
        transforms ``initial_state`` into a superset of ``goal_state``,
        or ``None`` if no plan exists.
        """
        initial = frozenset(initial_state)
        goal    = frozenset(goal_state)

        queue   = deque([(initial, [])])
        visited = {initial}

        while queue:
            state, plan = queue.popleft()
            if goal.issubset(state):
                return plan

            for action in self.action_library:
                new_state = self._apply_action(state, action)
                if new_state is not None and new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, plan + [action]))

        return None

    # ------------------------------------------------------------------
    # Diagnosis normalization & symptom-based inference
    # ------------------------------------------------------------------
    def _normalize_diagnosis(self, dx: str) -> str:
        """Map a raw diagnosis string to a canonical disease key."""
        dx_clean = (str(dx).lower()
                    .replace('_', '').replace('-', '').replace(' ', ''))
        if 'covid' in dx_clean:
            return 'covid19'
        if 'cardiac' in dx_clean or 'heart' in dx_clean:
            return 'cardiac_event'
        if 'meningitis' in dx_clean:
            return 'meningitis'
        if 'diabetes' in dx_clean:
            return 'diabetes'
        if 'dengue' in dx_clean:
            return 'dengue'
        if 'tuberculosis' in dx_clean or 'tb' in dx_clean:
            return 'tuberculosis'
        if 'pneumonia' in dx_clean:
            return 'pneumonia'
        if 'cold' in dx_clean:
            return 'common_cold'
        if 'flu' in dx_clean or 'influenza' in dx_clean:
            return 'flu'
        return dx_clean

    def _infer_diagnosis_from_symptoms(self, symptoms: List[str]) -> str:
        """Simple rule-based diagnosis inference from a symptom list.

        Used when the percept does not carry an explicit diagnosis
        (e.g. the planner is called directly with a PatientPercept).
        """
        s = {str(x).lower().replace(' ', '_') for x in symptoms}

        if 'loss_of_smell' in s:
            return 'covid19'
        if 'fever' in s and 'cough' in s and 'fatigue' in s:
            return 'covid19'
        if 'fever' in s and 'body_aches' in s:
            return 'flu'
        if 'fever' in s and 'rash' in s and 'joint_pain' in s:
            return 'dengue'
        if 'chest_pain' in s and 'shortness_of_breath' in s:
            return 'cardiac_event'
        if 'frequent_urination' in s and 'excessive_thirst' in s:
            return 'diabetes'
        if 'cough' in s and 'runny_nose' in s and 'sore_throat' in s:
            return 'common_cold'
        if 'cough' in s and 'shortness_of_breath' in s:
            return 'pneumonia'
        return 'common_cold'

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------
    def _build_initial_state(self, dx_key: str) -> Set[str]:
        """Map a canonical diagnosis to STRIPS initial-state predicates."""
        base = {'PATIENT_PRESENT'}

        diagnosis_states = {
            'flu':           {'VIRAL_INFECTION', 'MILD_ILLNESS'},
            'covid19':       {'COVID_SUSPECTED', 'CONTAGIOUS_DISEASE',
                              'VIRAL_INFECTION'},
            'cardiac_event': {'CARDIAC_SUSPECTED', 'EMERGENCY_CASE',
                              'ICU_AVAILABLE'},
            'dengue':        {'VIRAL_INFECTION', 'DEHYDRATION_RISK'},
            'meningitis':    {'BACTERIAL_INFECTION', 'EMERGENCY_CASE',
                              'ICU_AVAILABLE'},
            'tuberculosis':  {'BACTERIAL_INFECTION', 'CONTAGIOUS_DISEASE'},
            'diabetes':      {'MILD_ILLNESS'},
            'pneumonia':     {'VIRAL_INFECTION', 'RESPIRATORY_DISTRESS'},
            'common_cold':   {'VIRAL_INFECTION', 'MILD_ILLNESS'},
        }

        return base | diagnosis_states.get(dx_key, {'MILD_ILLNESS'})

    def _build_goal_state(self, urgency: str) -> Set[str]:
        """Goal state: treatment, monitoring, and follow-up are always required."""
        goal = {'TREATMENT_STARTED', 'VITALS_MONITORED', 'FOLLOWUP_SCHEDULED'}
        if str(urgency).upper() == 'CRITICAL':
            goal.add('PATIENT_IN_ICU')
        return goal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def create_treatment_plan(self, diagnosis: str,
                              urgency: str) -> Dict:
        """Generate a treatment plan for a given diagnosis and urgency."""
        dx_key         = self._normalize_diagnosis(diagnosis)
        initial_state  = self._build_initial_state(dx_key)
        goal_state     = self._build_goal_state(urgency)

        plan = self.generate_plan(initial_state, goal_state)

        if plan is None:
            return {'error': 'No plan found', 'plan': [], 'steps': 0,
                    'diagnosis': diagnosis, 'urgency': urgency,
                    'confidence': 0.0,
                    'summary': f"No treatment plan found for {diagnosis}"}

        plan_steps = [
            {
                'step':     i + 1,
                'action':   a['name'],
                'duration': a['duration'],
                'cost':     a['cost'],
            }
            for i, a in enumerate(plan)
        ]

        return {
            'diagnosis':       diagnosis,
            'urgency':         urgency,
            'steps':           len(plan_steps),
            'plan':            plan_steps,
            'total_duration':  ', '.join(a['duration'] for a in plan),
            'confidence':      0.95,
            'summary':         f"Generated {len(plan_steps)}-step "
                               f"treatment plan for {diagnosis}",
        }

    def analyze(self, percept: Union[Dict, Any]) -> Dict:
        """
        Module interface for the agent (handles dict and PatientPercept).

        When the percept carries no explicit diagnosis (the normal case
        when called from ``agent.think()``), the planner infers a
        diagnosis from the patient's symptoms before planning.
        """
        if isinstance(percept, dict):
            dx       = percept.get('diagnosis')
            urgency  = percept.get('urgency', 'MEDIUM')
            symptoms = percept.get('symptoms', [])
        else:
            dx       = getattr(percept, 'diagnosis', None)
            urgency  = getattr(percept, 'urgency', 'MEDIUM')
            symptoms = getattr(percept, 'symptoms', [])

        # Fall back to symptom-based inference when no usable diagnosis
        if not dx or str(dx).strip().lower() in ('', 'unknown',
                                                 'insufficient data'):
            dx = self._infer_diagnosis_from_symptoms(
                symptoms if isinstance(symptoms, list) else [])

        result = self.create_treatment_plan(dx, urgency)

        # Standard module-interface keys
        result['module'] = 'Treatment Planner'
        return result


if __name__ == "__main__":
    planner = TreatmentPlanner()

    print("=" * 55)
    print("  STRIPS Treatment Planner — Smoke Test")
    print("=" * 55)

    test_cases = [
        ("covid19", "HIGH"),
        ("flu", "MEDIUM"),
        ("cardiac_event", "CRITICAL"),
        ("common_cold", "LOW"),
        ("pneumonia", "HIGH"),
    ]

    for dx, urgency in test_cases:
        result = planner.create_treatment_plan(dx, urgency)
        print(f"\nDiagnosis: {dx} | Urgency: {urgency}")
        if 'error' in result:
            print(f"  -> {result['error']}")
            continue
        for step in result['plan']:
            print(f"  Step {step['step']}: {step['action']} "
                  f"[{step['duration']}]")
        print(f"  => {result['summary']}")

    # Module interface test (PatientPercept-like dict, no diagnosis key)
    print("\n" + "=" * 55)
    print("  analyze() interface test (symptom-based inference)")
    print("=" * 55)
    percept = {'symptoms': ['fever', 'cough', 'fatigue', 'loss_of_smell']}
    out = planner.analyze(percept)
    print(f"  Module       : {out.get('module')}")
    print(f"  Diagnosis    : {out.get('diagnosis')}")
    print(f"  Confidence   : {out.get('confidence')}")
    print(f"  Plan steps   : {out.get('steps')}")


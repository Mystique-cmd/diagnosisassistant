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
    """

    def __init__(self):
        # Action Domain Definition (STRIPS Operators)
        self.actions = {
            'IsolatePatient': {
                'preconds': ['CONTAGIOUS_DISEASE'],
                'add': ['PATIENT_ISOLATED'],
                'del': [],
                'duration': '14 days',
            },
            'OrderPCRTest': {
                'preconds': ['COVID_SUSPECTED'],
                'add': ['PCR_ORDERED'],
                'del': [],
                'duration': '24 hours',
            },
            'ReceivePCRResult': {
                'preconds': ['PCR_ORDERED'],
                'add': ['PCR_CONFIRMED'],
                'del': ['COVID_SUSPECTED'],
                'duration': '24 hours',
            },
            'PrescribeAntiviral': {
                'preconds': ['VIRAL_INFECTION'],
                'add': ['TREATMENT_STARTED'],
                'del': [],
                'duration': '10 minutes',
            },
            'AdministerOxygen': {
                'preconds': ['RESPIRATORY_DISTRESS'],
                'add': ['OXYGEN_ADMINISTERED'],
                'del': [],
                'duration': 'Immediate',
            },
            'PrescribeAntibiotics': {
                'preconds': ['BACTERIAL_INFECTION'],
                'add': ['TREATMENT_STARTED'],
                'del': [],
                'duration': '10 minutes',
            },
            'AdministerAspirin': {
                'preconds': ['CARDIAC_SUSPECTED'],
                'add': ['CARDIAC_INITIAL_CARE', 'TREATMENT_STARTED'],
                'del': [],
                'duration': 'Immediate',
            },
            'MonitorVitals': {
                'preconds': ['PATIENT_PRESENT'],
                'add': ['VITALS_MONITORED'],
                'del': [],
                'duration': 'Continuous',
            },
            'ScheduleFollowUp': {
                'preconds': ['TREATMENT_STARTED'],
                'add': ['FOLLOWUP_SCHEDULED'],
                'del': [],
                'duration': '5 minutes',
            },
            'AdmitToICU': {
                'preconds': ['EMERGENCY_CASE'],
                'add': ['PATIENT_IN_ICU'],
                'del': [],
                'duration': 'Immediate',
            },
            'DischargePatient': {
                'preconds': ['PLAN_COMPLETE', 'SYMPTOMS_RESOLVED'],
                'add': ['PATIENT_DISCHARGED'],
                'del': ['PLAN_COMPLETE'],
                'duration': '30 minutes',
                'cost': 0,
            },
        }

        # Build the STRIPS action library (list form) used by the BFS planner
        self.action_library = []
        for name, spec in self.actions.items():
            self.action_library.append({
                'name': name,
                'precond': set(spec.get('preconds', [])),
                'add': set(spec.get('add', [])),
                'del': set(spec.get('del', [])),
                'cost': spec.get('cost', 1),
                'duration': spec.get('duration', 'N/A'),
            })

    def _apply_action(self, state: frozenset,
                      action: Dict) -> Optional[frozenset]:
        """Return the successor state if the action is applicable, else None."""
        if not action['precond'].issubset(state):
            return None
        return frozenset((state - action['del']) | action['add'])

    def generate_plan(self,
                      initial_state: Set[str],
                      goal_state:    Set[str]) -> Optional[List[Dict]]:
        """BFS-based forward state-space plan generation."""
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

    def _normalize_diagnosis(self, dx: str) -> str:
        """Map a raw diagnosis string to a canonical disease key."""
        dx_clean = (str(dx).lower()
                    .replace('_', '').replace('-', '').replace(' ', ''))
        if 'covid' in dx_clean:
            return 'covid19'
        if 'cardiac' in dx_clean or 'heart' in dx_clean:
            return 'cardiac_event'
        if 'cold' in dx_clean:
            return 'common_cold'
        if 'flu' in dx_clean or 'influenza' in dx_clean:
            return 'flu'
        if 'pneumonia' in dx_clean:
            return 'pneumonia'
        if 'tuberculosis' in dx_clean or 'tb' in dx_clean:
            return 'tuberculosis'
        return dx_clean

    def create_treatment_plan(self, diagnosis: str,
                              urgency: str) -> Dict:
        """Generate a treatment plan for a given diagnosis and urgency."""
        dx_key = self._normalize_diagnosis(diagnosis)

        # Map diagnosis to initial state predicates
        diagnosis_states = {
            'flu':           {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'covid19':       {'COVID_SUSPECTED', 'CONTAGIOUS_DISEASE',
                              'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'cardiac_event': {'CARDIAC_SUSPECTED', 'EMERGENCY_CASE',
                              'ICU_AVAILABLE'},
            'dengue':        {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED',
                              'DEHYDRATION_RISK'},
            'meningitis':    {'BACTERIAL_INFECTION', 'EMERGENCY_CASE',
                              'ICU_AVAILABLE'},
            'tuberculosis':  {'BACTERIAL_INFECTION', 'CONTAGIOUS_DISEASE',
                              'DIAGNOSIS_NEEDED'},
            'diabetes':      {'DIAGNOSIS_NEEDED'},
            'pneumonia':     {'VIRAL_INFECTION', 'RESPIRATORY_DISTRESS',
                              'DIAGNOSIS_NEEDED'},
            'common_cold':   {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
        }

        base_state = {'PATIENT_PRESENT'}
        dx_state = diagnosis_states.get(dx_key, {'DIAGNOSIS_NEEDED'})
        initial_state = base_state | dx_state

        # Goal state: always end with treatment and monitoring
        goal_state = {'TREATMENT_STARTED', 'VITALS_MONITORED',
                      'FOLLOWUP_SCHEDULED'}
        if urgency == 'CRITICAL':
            goal_state.add('PATIENT_IN_ICU')

        plan = self.generate_plan(initial_state, goal_state)

        if plan is None:
            return {'error': 'No plan found', 'plan': [], 'steps': 0,
                    'diagnosis': diagnosis, 'urgency': urgency}

        plan_steps = [
            {
                'step':     i + 1,
                'action':   action['name'],
                'duration': action['duration'],
                'cost':     action['cost'],
            }
            for i, action in enumerate(plan)
        ]

        return {
            'diagnosis':      diagnosis,
            'urgency':        urgency,
            'steps':          len(plan_steps),
            'plan':           plan_steps,
            'total_duration': f"{len(plan_steps)} actions | see individual durations",
            'summary':        (f"{len(plan_steps)}-step treatment plan "
                               f"for {diagnosis} ({urgency} urgency)"),
        }

    def analyze(self, percept: Union[Dict, Any]) -> Dict:
        """Module interface for the agent (handles dict and PatientPercept objects)."""
        if isinstance(percept, dict):
            dx = percept.get('diagnosis', 'flu')
            urgency = percept.get('urgency', 'MEDIUM')
        else:
            dx = getattr(percept, 'diagnosis', 'flu')
            urgency = getattr(percept, 'urgency', 'MEDIUM')

        return self.create_treatment_plan(dx, urgency)


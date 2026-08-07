# ============================================================
# MODULE 8: Automated Planning — STRIPS Treatment Planner
# Covers: Week 14 (Automated Planning)
# ============================================================

from typing import Dict, List, Any, Union

class TreatmentPlanner:
    """
    STRIPS-based automated planner for generating patient treatment plans.
    """

    def __init__(self):
        # Action Domain Definition (STRIPS Operators)
        self.actions = {
            'IsolatePatient': {
                'preconds': ['CONTAGIOUS_DISEASE'],
                'add': ['PATIENT_ISOLATED'],
                'del': [],
                'duration': '14 days'
            },
            'OrderPCRTest': {
                'preconds': ['COVID_SUSPECTED'],
                'add': ['PCR_ORDERED'],
                'del': [],
                'duration': '24 hours'
            },
            'ReceivePCRResult': {
                'preconds': ['PCR_ORDERED'],
                'add': ['PCR_CONFIRMED'],
                'del': ['COVID_SUSPECTED'],
                'duration': '24 hours'
            },
            'PrescribeAntiviral': {
                'preconds': ['VIRAL_INFECTION'],
                'add': ['TREATMENT_STARTED'],
                'del': [],
                'duration': '10 minutes'
            },
            'AdministerOxygen': {
                'preconds': ['RESPIRATORY_DISTRESS'],
                'add': ['OXYGEN_ADMINISTERED'],
                'del': [],
                'duration': 'Immediate'
            },
            'PrescribeAntibiotics': {
                'preconds': ['BACTERIAL_INFECTION'],
                'add': ['TREATMENT_STARTED'],
                'del': [],
                'duration': '10 minutes'
            },
            'AdministerAspirin': {
                'preconds': ['CARDIAC_SUSPECTED'],
                'add': ['CARDIAC_INITIAL_CARE'],
                'del': [],
                'duration': 'Immediate'
            },
            'MonitorVitals': {
                'preconds': ['PATIENT_PRESENT'],
                'add': ['VITALS_MONITORED'],
                'del': [],
                'duration': 'Continuous'
            },
            'ScheduleFollowUp': {
                'preconds': ['TREATMENT_STARTED'],
                'add': ['FOLLOWUP_SCHEDULED'],
                'del': [],
                'duration': '5 minutes'
            },
<<<<<<< Updated upstream
            {
                'name': 'DischargePatient',
                'precond': {'PLAN_COMPLETE', 'SYMPTOMS_RESOLVED'},
                'delete':  {'PLAN_COMPLETE'},
                'add':     {'PATIENT_DISCHARGED'},
                'cost': 0, 'duration': '30 minutes'
            },
        ]

    def _apply_action(self, state: frozenset,
                      action: Dict) -> Optional[frozenset]:
        if not action['precond'].issubset(state):
            return None
        return frozenset((state - action['delete']) | action['add'])

    def generate_plan(self,
                      initial_state: Set[str],
                      goal_state:    Set[str]) -> Optional[List[Dict]]:
        """BFS-based plan generation"""
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
                if new_state and new_state not in visited:
                    visited.add(new_state)
                    queue.append((new_state, plan + [action]))

        return None

    def create_treatment_plan(self, diagnosis: str,
                              urgency: str) -> Dict:
        """Generate a treatment plan for a given diagnosis"""

        # Map diagnosis to initial state predicates
        diagnosis_states = {
            'flu':           {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
            'covid19':       {'COVID_SUSPECTED', 'CONTAGIOUS_DISEASE',
                              'DIAGNOSIS_NEEDED'},
            'cardiac_event': {'EMERGENCY_CASE',  'ICU_AVAILABLE'},
            'dengue':        {'VIRAL_INFECTION',  'DIAGNOSIS_NEEDED',
                              'DEHYDRATION_RISK'},
            'meningitis':    {'EMERGENCY_CASE',  'BACTERIAL_INFECTION',
                              'ICU_AVAILABLE'},
            'tuberculosis':  {'BACTERIAL_INFECTION', 'CONTAGIOUS_DISEASE',
                              'DIAGNOSIS_NEEDED'},
            'diabetes':      {'DIAGNOSIS_NEEDED'},
            'common_cold':   {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED'},
        }

        base_state = {'PATIENT_PRESENT'}
        dx_state   = diagnosis_states.get(
            diagnosis.lower().replace(' ', '_'),
            {'DIAGNOSIS_NEEDED'}
        )
        initial_state = base_state | dx_state

        # Goal state: always end with treatment and monitoring
        goal_state = {'TREATMENT_STARTED', 'VITALS_MONITORED',
                      'FOLLOWUP_SCHEDULED'}
        if urgency == 'CRITICAL':
            goal_state.add('PATIENT_IN_ICU')

        plan = self.generate_plan(initial_state, goal_state)

        if plan is None:
            return {'error': 'No plan found', 'plan': [], 'steps': 0}

        plan_steps = [
            {
                'step':     i+1,
                'action':   a['name'],
                'duration': a['duration'],
                'cost':     a['cost']
=======
            'RestAndHydrate': {
                'preconds': ['MILD_ILLNESS'],
                'add': ['TREATMENT_STARTED'],
                'del': [],
                'duration': '3-5 days'
>>>>>>> Stashed changes
            }
        }

    def _normalize_diagnosis(self, dx: str) -> str:
        dx_clean = str(dx).lower().replace('_', '').replace('-', '').replace(' ', '')
        if 'covid' in dx_clean: return 'covid19'
        if 'cardiac' in dx_clean or 'heart' in dx_clean: return 'cardiac_event'
        if 'cold' in dx_clean: return 'common_cold'
        if 'flu' in dx_clean or 'influenza' in dx_clean: return 'flu'
        if 'pneumonia' in dx_clean: return 'pneumonia'
        return dx_clean

<<<<<<< Updated upstream
    def analyze(self, percept) -> Dict:
        """Module interface — generates a treatment plan from patient percept"""
        if isinstance(percept, dict):
            dx = percept.get('diagnosis', 'flu')
            urgency = percept.get('urgency', 'MEDIUM')
        else:
            dx = getattr(percept, 'diagnosis', 'flu')
            urgency = getattr(percept, 'urgency', 'MEDIUM')
=======
    def generate_plan(self, diagnosis: str, urgency: str = "MODERATE") -> List[Dict[str, Any]]:
        dx_key = self._normalize_diagnosis(diagnosis)
>>>>>>> Stashed changes

        # Initial state setup based on normalized diagnosis
        initial_state = {'PATIENT_PRESENT'}
        
        if dx_key == 'covid19':
            initial_state.update({'CONTAGIOUS_DISEASE', 'COVID_SUSPECTED', 'VIRAL_INFECTION'})
        elif dx_key == 'cardiac_event':
            initial_state.update({'CARDIAC_SUSPECTED', 'RESPIRATORY_DISTRESS'})
        elif dx_key in ['common_cold', 'flu']:
            initial_state.update({'VIRAL_INFECTION', 'MILD_ILLNESS'})
        else:
            initial_state.update({'MILD_ILLNESS'})

        # Forward State-Space Search Planner
        current_state = set(initial_state)
        plan = []
        step_num = 1

        # Determine sequence of actions based on initial state predicates
        for action_name, details in self.actions.items():
            preconds = set(details['preconds'])
            if preconds.issubset(current_state):
                plan.append({
                    'step': step_num,
                    'action': action_name,
                    'duration': details['duration']
                })
                # State transition
                current_state.update(details['add'])
                current_state.difference_update(details['del'])
                step_num += 1

        return plan

    def analyze(self, percept: Union[Dict, Any]) -> Dict:
        """Module interface for the agent (handles dict and object inputs)"""
        if isinstance(percept, dict):
            dx = percept.get('diagnosis', 'common_cold')
            urgency = percept.get('urgency', 'MODERATE')
        else:
            dx = getattr(percept, 'diagnosis', 'common_cold')
            urgency = getattr(percept, 'urgency', 'MODERATE')

        plan_steps = self.generate_plan(dx, urgency)

        return {
            'diagnosis': dx,
            'urgency': urgency,
            'steps': len(plan_steps),
            'total_duration': f"{len(plan_steps)} actions | see individual durations",
            'plan': plan_steps,
            'summary': f"Generated {len(plan_steps)}-step treatment plan for {dx}"
        }

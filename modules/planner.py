from copy import deepcopy
from collections import deque
from typing import Dict, List, Set, Tuple, Optional


class TreatmentPlanner:
    """
    STRIPS-based treatment planner.
    Generates step-by-step treatment plans
    from patient diagnosis to recovery.
    """

    def __init__(self):
        self.action_library = self._build_action_library()

    def _build_action_library(self) -> List[Dict]:
        """Define medical treatment actions"""
        return [
            # Emergency Actions
            {
                'name': 'CallEmergencyServices',
                'precond': {'EMERGENCY_CASE', 'PATIENT_PRESENT'},
                'delete':  {'EMERGENCY_CASE'},
                'add':     {'EMERGENCY_SERVICES_CALLED'},
                'cost': 0, 'duration': '5 minutes'
            },
            {
                'name': 'TransferToICU',
                'precond': {'EMERGENCY_SERVICES_CALLED', 'ICU_AVAILABLE'},
                'delete':  {'EMERGENCY_SERVICES_CALLED'},
                'add':     {'PATIENT_IN_ICU', 'MONITORING_ACTIVE'},
                'cost': 0, 'duration': '15 minutes'
            },
            {
               
                'name': 'AdministerCardiacCare',
                'precond': {'PATIENT_IN_ICU', 'CARDIAC_CONDITION'},
                'delete':  {'CARDIAC_CONDITION'},
                'add':     {'TREATMENT_STARTED'},
                'cost': 1, 'duration': '20 minutes'
            },
            {
               
                'name': 'IsolatePatient',
                'precond': {'CONTAGIOUS_DISEASE', 'PATIENT_PRESENT'},
                'delete':  {'CONTAGIOUS_DISEASE'},
                'add':     {'PATIENT_ISOLATED'},
                'cost': 0, 'duration': '14 days'
            },
            # Diagnostics
            {
                'name': 'OrderBloodPanel',
                'precond': {'PATIENT_PRESENT', 'DIAGNOSIS_NEEDED'},
                'delete':  {'DIAGNOSIS_NEEDED'},
                'add':     {'BLOOD_RESULTS_PENDING'},
                'cost': 1, 'duration': '30 minutes'
            },
            {
                'name': 'ReceiveBloodResults',
                
                'precond': {'BLOOD_RESULTS_PENDING'},
                'delete':  {'BLOOD_RESULTS_PENDING'},
                'add':     {'BLOOD_RESULTS_AVAILABLE', 'DIAGNOSIS_REFINED',
                            'DIAGNOSIS_CONFIRMED'},
                'cost': 0, 'duration': '2 hours'
            },
            {
                'name': 'OrderPCRTest',
                'precond': {'COVID_SUSPECTED', 'PATIENT_PRESENT'},
                'delete':  {'COVID_SUSPECTED'},
                'add':     {'PCR_PENDING'},
                'cost': 1, 'duration': '24 hours'
            },
            {
                'name': 'ReceivePCRResult',
                'precond': {'PCR_PENDING'},
                'delete':  {'PCR_PENDING'},
                'add':     {'PCR_RESULT_AVAILABLE', 'DIAGNOSIS_CONFIRMED'},
                'cost': 0, 'duration': '24 hours'
            },
            # Treatment
            {
                
                'name': 'PrescribeAntiviral',
                'precond': {'DIAGNOSIS_CONFIRMED', 'VIRAL_INFECTION',
                            'PATIENT_ISOLATED'},
                'delete':  {'VIRAL_INFECTION'},
                'add':     {'ANTIVIRAL_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
               
                'name': 'PrescribeAntiviralOutpatient',
                'precond': {'DIAGNOSIS_CONFIRMED', 'VIRAL_INFECTION',
                            'NON_CONTAGIOUS_CASE'},
                'delete':  {'VIRAL_INFECTION'},
                'add':     {'ANTIVIRAL_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                
                'name': 'PrescribeAntibiotics',
                'precond': {'DIAGNOSIS_CONFIRMED', 'BACTERIAL_INFECTION',
                            'PATIENT_ISOLATED'},
                'delete':  {'BACTERIAL_INFECTION'},
                'add':     {'ANTIBIOTICS_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
              
                'name': 'PrescribeAntibioticsOutpatient',
                'precond': {'DIAGNOSIS_CONFIRMED', 'BACTERIAL_INFECTION',
                            'NON_CONTAGIOUS_CASE'},
                'delete':  {'BACTERIAL_INFECTION'},
                'add':     {'ANTIBIOTICS_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '10 minutes'
            },
            {
                
                'name': 'PrescribeInsulinTherapy',
                'precond': {'DIAGNOSIS_CONFIRMED', 'METABOLIC_CONDITION'},
                'delete':  {'METABOLIC_CONDITION'},
                'add':     {'INSULIN_PRESCRIBED', 'TREATMENT_STARTED'},
                'cost': 1, 'duration': '15 minutes'
            },
            {
                'name': 'AdministerFluids',
                'precond': {'PATIENT_IN_ICU', 'DEHYDRATION_RISK'},
                'delete':  {'DEHYDRATION_RISK'},
                'add':     {'FLUIDS_ADMINISTERED'},
                'cost': 1, 'duration': '1 hour'
            },
            {
                'name': 'MonitorVitals',
                'precond': {'TREATMENT_STARTED', 'PATIENT_PRESENT'},
                'delete':  set(),
                'add':     {'VITALS_MONITORED'},
                'cost': 0, 'duration': 'Continuous'
            },
            {
                'name': 'ScheduleFollowUp',
                'precond': {'TREATMENT_STARTED', 'VITALS_MONITORED'},
                'delete':  set(),
                'add':     {'FOLLOWUP_SCHEDULED', 'PLAN_COMPLETE'},
                'cost': 0, 'duration': '5 minutes'
            },
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

       
        diagnosis_states = {
            'flu':           {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED',
                              'NON_CONTAGIOUS_CASE'},
            'covid19':       {'VIRAL_INFECTION', 'COVID_SUSPECTED',
                              'CONTAGIOUS_DISEASE'},
            'cardiac_event': {'EMERGENCY_CASE', 'ICU_AVAILABLE',
                              'CARDIAC_CONDITION'},
            'dengue':        {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED',
                              'DEHYDRATION_RISK', 'NON_CONTAGIOUS_CASE'},
            'meningitis':    {'EMERGENCY_CASE', 'BACTERIAL_INFECTION',
                              'ICU_AVAILABLE', 'DIAGNOSIS_NEEDED',
                              'NON_CONTAGIOUS_CASE'},
            'tuberculosis':  {'BACTERIAL_INFECTION', 'CONTAGIOUS_DISEASE',
                              'DIAGNOSIS_NEEDED'},
            'diabetes':      {'DIAGNOSIS_NEEDED', 'METABOLIC_CONDITION'},
            'common_cold':   {'VIRAL_INFECTION', 'DIAGNOSIS_NEEDED',
                              'NON_CONTAGIOUS_CASE'},
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
            
            return {
                'diagnosis': diagnosis,
                'urgency': urgency,
                'initial_state': sorted(initial_state),
                'goal_state': sorted(goal_state),
                'error': 'No plan found',
                'steps': 0,
                'plan': []
            }

        return {
            'diagnosis':     diagnosis,
            'urgency':       urgency,
            'initial_state': sorted(initial_state),
            'goal_state':    sorted(goal_state),
            'steps':         len(plan),
            'total_duration': self._estimate_duration(plan),
            'plan': [
                {
                    'step':     i+1,
                    'action':   a['name'],
                    'duration': a['duration'],
                    'cost':     a['cost']
                }
                for i, a in enumerate(plan)
            ]
        }

    def _estimate_duration(self, plan: List[Dict]) -> str:
        durations = [a['duration'] for a in plan]
        return f"{len(plan)} actions | see individual durations"

    def analyze(self, percept) -> Dict:
        """Module interface — generates a plan from the percept"""
       
        diagnosis = getattr(percept, 'diagnosis', 'flu')
        urgency   = getattr(percept, 'urgency', 'MEDIUM')

        result = self.create_treatment_plan(diagnosis, urgency)
        result['summary']    = f"Plan: {result.get('steps', 0)} steps generated"
        result['diagnosis']  = diagnosis
        result['confidence'] = getattr(percept, 'confidence', 0.7)
        return result



# Manual test

if __name__ == "__main__":
    planner = TreatmentPlanner()

    # Test for COVID-19 case
    plan = planner.create_treatment_plan('covid19', 'HIGH')
    print(f"Diagnosis : {plan['diagnosis']}")
    print(f"Plan Steps: {plan['steps']}")
    print()
    for step in plan['plan']:
        print(f"  Step {step['step']:2d}: {step['action']:<30} [{step['duration']}]")

    print("\n" + "=" * 55)
    print("Other diagnoses (MEDIUM urgency):")
    print("=" * 55)
    for dx in ['flu', 'cardiac_event', 'dengue', 'meningitis',
               'tuberculosis', 'diabetes', 'common_cold']:
        p = planner.create_treatment_plan(dx, 'MEDIUM')
        if p.get('error'):
            print(f"\n{dx} -> ERROR: {p['error']}")
        else:
            print(f"\n{dx} -> {p['steps']} steps: "
                  f"{[s['action'] for s in p['plan']]}")

    print("\n" + "=" * 55)
    print("meningitis at CRITICAL urgency (must route through ICU):")
    print("=" * 55)
    crit = planner.create_treatment_plan('meningitis', 'CRITICAL')
    for step in crit['plan']:
        print(f"  Step {step['step']:2d}: {step['action']:<30} [{step['duration']}]")

    print("\n" + "=" * 55)
    print("analyze(percept) interface:")
    print("=" * 55)

    class FakePercept:
        def __init__(self, diagnosis, urgency, confidence=0.9):
            self.diagnosis = diagnosis
            self.urgency = urgency
            self.confidence = confidence

    analysis = planner.analyze(FakePercept('tuberculosis', 'MEDIUM'))
    print(analysis['summary'])
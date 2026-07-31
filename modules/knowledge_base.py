"""
Module 2: Medical Knowledge Base & Inference Engine
Uses First-Order Logic (FOL), Forward Chaining, and Backward Chaining.
"""

class MedicalKnowledgeBase:
    def __init__(self):
        # Step 2: Facts stored in internal dict/list with certainty factors
        self.facts = {}
        
        # Step 1 & 2: Rules stored as tuples: (list_of_conditions, conclusion, certainty_factor)
        self.rules = []
        self._load_default_rules()

    def add_fact(self, fact_name: str, certainty: float = 1.0):
        """Step 2: Stores data in KB internal facts with certainty factor."""
        # Step 3 fix: Clean string to lower_snake_case
        clean_fact = fact_name.lower().strip().replace(" ", "_")
        self.facts[clean_fact] = float(certainty)

    def add_rule(self, conditions: list, conclusion: str, cf: float = 1.0):
        """Step 2: Stores rules as tuples: (conditions, conclusion, cf)."""
        clean_conditions = [c.lower().strip().replace(" ", "_") for c in conditions]
        clean_conclusion = conclusion.lower().strip().replace(" ", "_")
        # Tuples according to Step 1 specification
        self.rules.append((clean_conditions, clean_conclusion, float(cf)))

    def clear_facts(self):
        """Resets facts for a new patient evaluation."""
        self.facts.clear()

    def _load_default_rules(self):
        """Default rules matching the manual."""
        self.add_rule(["fever", "cough", "loss_of_smell", "fatigue"], "covid19_suspected", 0.85)
        self.add_rule(["fever", "cough", "fatigue"], "flu_suspected", 0.75)
        self.add_rule(["fever", "rash", "joint_pain"], "dengue_suspected", 0.80)
        self.add_rule(["cough", "runny_nose", "sore_throat"], "common_cold_suspected", 0.70)
        self.add_rule(["chest_pain", "shortness_of_breath"], "cardiac_event_suspected", 0.90)

    def load_patient_symptoms(self, symptoms: list, temperature: float = 37.0):
        """Step 3 & Step 6: Translates patient symptoms and vitals into KB facts."""
        for symptom in symptoms:
            self.add_fact(symptom, 1.0)
            
        # Step 6: Add vitals (fever = temperature > 38°C)
        if temperature > 38.0:
            self.add_fact("fever", 1.0)

    def forward_chain(self, verbose: bool = False) -> dict:
        """Step 4: Loop-based Forward Chaining."""
        inferred = {}
        added = True
        iteration = 1

        while added:
            added = False
            for conditions, conclusion, cf in self.rules:
                # If ALL conditions are known facts
                if all(cond in self.facts for cond in conditions):
                    # Prevent re-inferring known facts (prevents infinite loops)
                    if conclusion not in self.facts:
                        # Combine CFs: rule_CF * min(condition_CFs)
                        min_cond_cf = min(self.facts[cond] for cond in conditions)
                        calculated_cf = round(cf * min_cond_cf, 3)
                        
                        self.facts[conclusion] = calculated_cf
                        inferred[conclusion] = calculated_cf
                        added = True
                        
                        if verbose:
                            cond_str = " ^ ".join(conditions)
                            print(f"Iter {iteration}: {cond_str} => {conclusion} (CF={calculated_cf:.3f})")
            iteration += 1

        return inferred

    def backward_chain(self, goal: str, visited=None) -> tuple[bool, float]:
        """Step 5: Recursive Backward Chaining."""
        goal = goal.lower().strip().replace(" ", "_")
        if visited is None:
            visited = set()

        if goal in visited:
            return False, 0.0
        visited.add(goal)

        # Base case: If GOAL is already a known fact -> return True
        if goal in self.facts:
            return True, self.facts[goal]

        # For each rule whose conclusion == GOAL
        matching_rules = [r for r in self.rules if r[1] == goal]
        
        for conditions, conclusion, cf in matching_rules:
            all_conditions_proven = True
            cond_cfs = []

            # Try to prove ALL conditions of that rule
            for cond in conditions:
                proven, cond_cf = self.backward_chain(cond, visited.copy())
                if proven:
                    cond_cfs.append(cond_cf)
                else:
                    all_conditions_proven = False
                    break

            # If successful -> GOAL is proved
            if all_conditions_proven and cond_cfs:
                overall_cf = round(cf * min(cond_cfs), 3)
                return True, overall_cf

        return False, 0.0

    def analyze(self, patient) -> dict:
        """Step 6: Standard interface for Intelligent Agent."""
        # 1. Clear old facts
        self.clear_facts()
        
        symptoms = getattr(patient, 'symptoms', [])
        temp = getattr(patient, 'temperature', 37.0)
        
        # 2 & 3. Load patient symptoms & add vitals
        self.load_patient_symptoms(symptoms, temp)
        
        # 4. Run forward chaining
        inferred = self.forward_chain()
        
        # 5. Return the top diagnosis and confidence
        if inferred:
            top_diag = max(inferred.items(), key=lambda x: x[1])
            return {
                'module': 'Knowledge Base',
                'diagnosis': top_diag[0].replace("_suspected", ""),
                'confidence': top_diag[1],
                'all_inferred': inferred
            }
            
        return {
            'module': 'Knowledge Base',
            'diagnosis': 'unknown',
            'confidence': 0.0,
            'all_inferred': {}
        }


# Step 7: How To Test This Module
if __name__ == "__main__":
    kb = MedicalKnowledgeBase()
    kb.add_fact("fever")
    kb.add_fact("cough")
    kb.add_fact("loss_of_smell")
    kb.add_fact("fatigue")
    
    results = kb.forward_chain(verbose=True)
    print("Inferred:", results)
    
    # Test backward chaining
    proved, cf = kb.backward_chain("covid19_suspected")
    print(f"COVID-19 suspected: {proved}, Confidence: {cf}")
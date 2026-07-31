# ============================================================
# MODULE — Search Algorithms
# Covers: Week 3 (Uninformed & Informed Search)
#
# NOTE: This module is not covered in the official lab manual
# (search.py is listed in the project structure but has no
# dedicated instructions). This is a reasonable implementation
# based on standard Week 3 search-algorithm content, designed to
# plug into the same agent.analyze() interface as every other
# module. Confirm with your instructor/team before treating this
# as final.
# ============================================================

import heapq
from collections import deque
from typing import Dict, List, Tuple, Optional


class MedicalSearchEngine:
    """
    Demonstrates classic search algorithms (BFS, DFS, UCS, A*)
    applied to medical diagnosis, modeled as a weighted graph
    search problem.

    Graph structure (built per-query, since it depends on which
    symptoms the patient actually has):

        START --(cost 0)--> (disease, 0)   for every candidate disease
        (disease, k) --(cost = 1 - P(symptom_k|disease))--> (disease, k+1)

    Each candidate disease gets its own evidence CHAIN, one link
    per matched symptom. Reaching (disease, N) — where N is the
    number of matched symptoms — means the search has weighed
    ALL of the patient's symptoms as evidence for that specific
    disease. The total path cost is therefore the full aggregate
    cost of that hypothesis, not just a single symptom's strength.

    An edge cost of a symptom-disease link represents how
    *unlikely* that symptom is to indicate that disease — a
    strong, textbook symptom (e.g. fever -> dengue, P=0.98) is a
    *cheap* edge (cost 0.02), while a weak/rare symptom is an
    *expensive* edge (cost close to 1.0).

    Finding the diagnosis becomes a shortest-path problem: the
    disease chain reached with the LOWEST total cost is the most
    probable diagnosis given ALL the evidence.

    Teaching point: every chain has the same length (number of
    matched symptoms), so BFS/DFS will find a *complete* path but
    stop at whichever disease happens to be explored/ordered
    first — NOT necessarily the cheapest one. UCS (uniform cost
    search / Dijkstra) and A* are the ones guaranteed to find the
    optimal (cheapest / most-confident) diagnosis by actually
    comparing total accumulated cost.
    """

    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'headache',
        'body_aches', 'loss_of_smell', 'chest_pain',
        'rash', 'joint_pain', 'shortness_of_breath',
        'sweating', 'frequent_urination', 'excessive_thirst',
        'blurred_vision', 'night_sweats', 'weight_loss',
        'stiff_neck', 'light_sensitivity'
    ]

    DISEASE_LABELS = [
        'flu', 'covid19', 'dengue', 'cardiac_event',
        'diabetes', 'common_cold', 'tuberculosis', 'meningitis'
    ]

    # Same disease -> symptom probability profiles used in
    # neural_network.py's synthetic data generator, so every
    # module in this system agrees on what each disease "looks
    # like".
    PROFILES = {
        'flu':           {'fever': 0.90, 'cough': 0.85, 'fatigue': 0.88,
                           'headache': 0.70, 'body_aches': 0.80},
        'covid19':       {'fever': 0.88, 'cough': 0.80, 'fatigue': 0.90,
                           'loss_of_smell': 0.85, 'headache': 0.65},
        'dengue':        {'fever': 0.98, 'rash': 0.75, 'joint_pain': 0.85,
                           'headache': 0.90, 'fatigue': 0.80},
        'cardiac_event': {'chest_pain': 0.92, 'shortness_of_breath': 0.88,
                           'sweating': 0.75, 'fatigue': 0.70},
        'diabetes':      {'fatigue': 0.82, 'frequent_urination': 0.95,
                           'excessive_thirst': 0.92, 'blurred_vision': 0.70},
        'common_cold':   {'cough': 0.90, 'fever': 0.50, 'headache': 0.60,
                           'fatigue': 0.55},
        'tuberculosis':  {'cough': 0.95, 'weight_loss': 0.85,
                           'night_sweats': 0.80, 'fatigue': 0.88,
                           'fever': 0.70},
        'meningitis':    {'headache': 0.95, 'stiff_neck': 0.90,
                           'fever': 0.92, 'light_sensitivity': 0.85},
    }

    # Cheapest possible single edge anywhere in the system — used as
    # a global lower bound for the A* heuristic (still admissible,
    # since no real edge can ever cost less than this).
    GLOBAL_MIN_COST = round(1.0 - 0.98, 4)  # strongest link in PROFILES

    START = ('START',)

    def __init__(self):
        pass  # graph is built per-query in _build_query_graph()

    # ------------------------------------------------------------------
    # Graph construction (per query — depends on matched symptoms)
    # ------------------------------------------------------------------
    def _clean(self, symptoms: List[str]) -> List[str]:
        cleaned = [s.lower().replace(' ', '_') for s in symptoms]
        return [s for s in cleaned if s in self.SYMPTOM_FEATURES]

    def _build_query_graph(self, matched_symptoms: List[str]) -> Dict[Tuple, List[Tuple[Tuple, float]]]:
        """
        Build one evidence chain per candidate disease:
            START -> (disease, 0) -> (disease, 1) -> ... -> (disease, N)
        where N = len(matched_symptoms). Reaching (disease, N) means
        every matched symptom has been weighed as evidence for that
        disease — this is the goal state for that branch.
        """
        n = len(matched_symptoms)
        graph: Dict[Tuple, List[Tuple[Tuple, float]]] = {self.START: []}

        for disease in self.DISEASE_LABELS:
            profile = self.PROFILES[disease]
            graph[self.START].append(((disease, 0), 0.0))
            for k in range(n):
                symptom = matched_symptoms[k]
                p = profile.get(symptom, self.BASELINE_P)
                cost = round(1.0 - p, 4)
                node, nxt = (disease, k), (disease, k + 1)
                graph.setdefault(node, []).append((nxt, cost))
            graph.setdefault((disease, n), [])  # goal node, no outgoing edges

        return graph

    BASELINE_P = 0.03  # default P(symptom|disease) if not listed above

    def _heuristic(self, node: Tuple, total_steps: int) -> float:
        """
        h(n): admissible lower bound on remaining cost = however many
        symptom-links are still left to check, times the cheapest
        possible link cost anywhere in the system.
        """
        if node == self.START:
            remaining = total_steps
        else:
            _, k = node
            remaining = total_steps - k
        return remaining * self.GLOBAL_MIN_COST

    def _is_goal(self, node: Tuple, total_steps: int) -> bool:
        return node != self.START and node[1] == total_steps

    # ------------------------------------------------------------------
    # Uninformed search
    # ------------------------------------------------------------------
    def bfs(self, symptoms: List[str]) -> Dict:
        """Breadth-First Search — fewest edges, ignores edge cost."""
        matched = self._clean(symptoms)
        if not matched:
            return self._empty_result('BFS')
        graph = self._build_query_graph(matched)
        n = len(matched)

        queue = deque([(self.START, [self.START], 0.0)])
        visited = {self.START}
        nodes_expanded = 0

        while queue:
            node, path, cost = queue.popleft()
            nodes_expanded += 1

            if self._is_goal(node, n):
                return self._format_result('BFS', path, cost, nodes_expanded, matched)

            for neighbor, edge_cost in graph.get(node, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor], cost + edge_cost))

        return self._empty_result('BFS')

    def dfs(self, symptoms: List[str]) -> Dict:
        """Depth-First Search — explores one branch fully before backtracking."""
        matched = self._clean(symptoms)
        if not matched:
            return self._empty_result('DFS')
        graph = self._build_query_graph(matched)
        n = len(matched)

        stack = [(self.START, [self.START], 0.0)]
        visited = set()
        nodes_expanded = 0

        while stack:
            node, path, cost = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            nodes_expanded += 1

            if self._is_goal(node, n):
                return self._format_result('DFS', path, cost, nodes_expanded, matched)

            for neighbor, edge_cost in reversed(graph.get(node, [])):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor], cost + edge_cost))

        return self._empty_result('DFS')

    # ------------------------------------------------------------------
    # Informed / optimal search
    # ------------------------------------------------------------------
    def uniform_cost_search(self, symptoms: List[str]) -> Dict:
        """UCS (Dijkstra) — guarantees the cheapest (most confident) path."""
        matched = self._clean(symptoms)
        if not matched:
            return self._empty_result('UCS')
        graph = self._build_query_graph(matched)
        n = len(matched)

        counter = 0  # tie-breaker so heapq never compares path lists
        frontier = [(0.0, counter, self.START, [self.START])]
        best_cost = {self.START: 0.0}
        nodes_expanded = 0

        while frontier:
            cost, _, node, path = heapq.heappop(frontier)
            nodes_expanded += 1

            if self._is_goal(node, n):
                return self._format_result('UCS', path, cost, nodes_expanded, matched)

            for neighbor, edge_cost in graph.get(node, []):
                new_cost = cost + edge_cost
                if new_cost < best_cost.get(neighbor, float('inf')):
                    best_cost[neighbor] = new_cost
                    counter += 1
                    heapq.heappush(frontier, (new_cost, counter, neighbor, path + [neighbor]))

        return self._empty_result('UCS')

    def a_star_search(self, symptoms: List[str]) -> Dict:
        """A* — UCS guided by a heuristic, same optimal result, fewer expansions."""
        matched = self._clean(symptoms)
        if not matched:
            return self._empty_result('A*')
        graph = self._build_query_graph(matched)
        n = len(matched)

        counter = 0
        frontier = [(self._heuristic(self.START, n), 0.0, counter, self.START, [self.START])]
        best_cost = {self.START: 0.0}
        nodes_expanded = 0

        while frontier:
            _, cost, _, node, path = heapq.heappop(frontier)
            nodes_expanded += 1

            if self._is_goal(node, n):
                return self._format_result('A*', path, cost, nodes_expanded, matched)

            for neighbor, edge_cost in graph.get(node, []):
                new_cost = cost + edge_cost
                if new_cost < best_cost.get(neighbor, float('inf')):
                    best_cost[neighbor] = new_cost
                    counter += 1
                    f = new_cost + self._heuristic(neighbor, n)
                    heapq.heappush(frontier, (f, new_cost, counter, neighbor, path + [neighbor]))

        return self._empty_result('A*')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _format_result(self, algorithm, path, cost, nodes_expanded, matched_symptoms) -> Dict:
        goal_node = path[-1]
        diagnosis = goal_node[0]
        # Rebuild a human-readable trail: disease -> symptom1 -> symptom2 -> ...
        readable_path = [diagnosis] + list(matched_symptoms)
        avg_cost = cost / len(matched_symptoms) if matched_symptoms else cost
        return {
            'algorithm':      algorithm,
            'diagnosis':      diagnosis,
            'path':           readable_path,
            'cost':           round(cost, 4),
            'confidence':     round(max(0.0, 1.0 - avg_cost), 4),
            'nodes_expanded': nodes_expanded,
        }

    def _empty_result(self, algorithm) -> Dict:
        return {
            'algorithm':      algorithm,
            'diagnosis':      'Unknown',
            'path':           [],
            'cost':           float('inf'),
            'confidence':     0.0,
            'nodes_expanded': 0,
        }

    def compare_algorithms(self, symptoms: List[str]) -> Dict[str, Dict]:
        """Run all four algorithms on the same symptoms and compare results."""
        results = {
            'BFS': self.bfs(symptoms),
            'DFS': self.dfs(symptoms),
            'UCS': self.uniform_cost_search(symptoms),
            'A*':  self.a_star_search(symptoms),
        }
        print(f"{'Algorithm':<6} {'Diagnosis':<16} {'Cost':>8} "
              f"{'Confidence':>11} {'Nodes Expanded':>15}")
        print("-" * 62)
        for name, r in results.items():
            print(f"{name:<6} {r['diagnosis']:<16} {r['cost']:>8.4f} "
                  f"{r['confidence']:>10.2%} {r['nodes_expanded']:>15}")
        return results

    def analyze(self, percept) -> Dict:
        """
        Module interface for the agent.
        Uses UCS as the authoritative result since it's guaranteed
        optimal, but reports what BFS/DFS would have found too —
        useful for the report's algorithm-comparison section.
        """
        ucs_result = self.uniform_cost_search(percept.symptoms)
        bfs_result = self.bfs(percept.symptoms)

        return {
            'summary':    f"Search: {ucs_result['diagnosis']} "
                          f"(optimal path cost={ucs_result['cost']:.3f})",
            'diagnosis':  ucs_result['diagnosis'],
            'confidence': ucs_result['confidence'],
            'path':       ucs_result['path'],
            'nodes_expanded': ucs_result['nodes_expanded'],
            'bfs_agreed': bfs_result['diagnosis'] == ucs_result['diagnosis'],
        }


if __name__ == "__main__":
    # ---- How To Test This Module ----
    engine = MedicalSearchEngine()

    test_symptoms = ["fever", "rash", "joint_pain", "headache"]  # dengue-like

    print("=" * 62)
    print("  Search Algorithms — Diagnosis via Graph Search")
    print(f"  Symptoms: {test_symptoms}")
    print("=" * 62)
    engine.compare_algorithms(test_symptoms)

    print("\nOptimal (UCS) path explanation:")
    ucs = engine.uniform_cost_search(test_symptoms)
    print(f"  {' -> '.join(ucs['path'])}")
    print(f"  Diagnosis : {ucs['diagnosis']}")
    print(f"  Confidence: {ucs['confidence']:.2%}")

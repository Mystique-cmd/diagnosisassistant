# TODO — Fix `modules/planner.py` SyntaxError

## Problem
`modules/planner.py` contains unresolved Git merge conflict markers
(`<<<<<<< Updated upstream`, `=======`, `>>>>>>> Stashed changes`) from a
`git stash` merge, causing a Python `SyntaxError` when `app.py` imports it.

## Steps
- [ ] Step 1: Rewrite `modules/planner.py` — remove all conflict markers,
      merge the upstream (BFS-based) and stashed branches into one clean
      STRIPS planner with a uniform action schema.
- [ ] Step 2: Verify the file compiles with `python -m py_compile modules/planner.py`.
- [ ] Step 3: Smoke-test import + `analyze()` without launching the GUI:
      `python -c "from modules.planner import TreatmentPlanner; ..."`.
- [ ] Step 4: Confirm `app.py` imports cleanly (GUI launch left to the user).


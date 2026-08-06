# TODO - GUI Implementation for Healthcare Diagnostic Assistant

## Steps
- [x] 1. Analyze codebase (app.py, modules, data) to understand the diagnosis pipeline
- [x] 2. Confirm GUI approach with user (Tkinter)
- [x] 3. Rewrite app.py with:
  - [x] a. Define available symptom options (from data/bayesian_net)
  - [x] b. Build DiagnosticGUI class (Tkinter) with patient info + symptom checkboxes
  - [x] c. Add "Run Diagnosis" button that builds PatientPercept and runs agent cycle
  - [x] d. Display color-coded diagnostic report (diagnosis, confidence, urgency, plan, recommendations)
  - [x] e. Handle invalid input with friendly error messages
- [x] 4. Optionally document tkinter in requirements.txt
- [x] 5. Test application compiles and parses correctly (py_compile / AST parse)

# Intelligent Healthcare Diagnostic Assistant

A Capstone project that integrates multiple AI diagnostic modules to assess a patient's health condition and recommend treatment based on the diagnosis made. The system modules are:
- Medical knowledge base
- Bayesian diagnostic inference
- Machine learning classifier
- Deep neural network model
- Fuzzy logic severity assessor
- AI treatment planner
- Tkinter GUI for patient vitals input and treatment recommendation

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

The app launches a GUI where users enter vitals, select symptoms, and run the diagnosis.

## Modules

- `modules/agent.py` — core agent orchestration
- `modules/knowledge_base.py` — medical knowledge retrieval
- `modules/bayesian_net.py` — Bayesian network diagnosis
- `modules/ml_classifier.py` — ML-based diagnostic model
- `modules/neural_network.py` — deep learning model
- `modules/fuzzy_controller.py` — fuzzy severity assessment
- `modules/planner.py` — treatment planning

## Requirements

See `requirements.txt` for required packages.

## Notes

- `tkinter` is part of the Python standard library.
- The app is designed for educational/demo use only, not actual clinical diagnosis.


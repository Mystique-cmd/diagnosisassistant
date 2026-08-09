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

PREREQUISITES: Install Python 3.9+, Git, VS Code editor/Jupyter notebook and pip first.

```bash
git clone [https://github.com/Mystique-cmd/diagnosisassistant.git](https://github.com/Mystique-cmd/diagnosisassistant.git)
cd diagnosisassistant

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

## Project Structure

diagnosisassistant/
│
├── data/
│   ├── final_training_data.csv # Merged 18-feature symptoms, diseases & patient_record dataset
|   └── test_data.csv           # Testing Dataset
|
├── evaluation/
|   ├── metric.py
|   └── visualizations.py
|
├── modules/
│   ├── agent.py               # Intelligent Core 'Perceive-Think-Act' Agent
│   ├── bayesian_net.py        # Probabilistic Calculations 
│   ├── fuzzy_controller.py    # Severity Assessment Logic
│   ├── knowledge_base.py      # FOL Inference Engine
│   ├── ml_classifier.py       # ML-based Diagnostic Model
│   ├── neural_network.py      # Deep Learning Model
│   ├── nlp_processor.py       # NLP + Text Processing
|   ├── planner.py             # STRIPS-based Treatment Planning Generator
|   ├── rl_agent.py            # Reinforcement Learning
│   └── search.py              # Search Algorithms
|
├── reports/
│   └── final_report.pdf
|                     
├── app.py                     # Main Application Entry
├── README.md
└── requirements.txt           # Required packages and dependencies

## Notes

- `tkinter` is part of the Python standard library.
- The app is designed for educational/demo use only, not actual clinical diagnosis.
- If TensorFlow fails to install due to path length limitations, ensure Long Paths are enabled in your Windows Registry.

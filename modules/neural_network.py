# ============================================================
# MODULE 5: Deep Neural Network Diagnostic Model
# Covers: Week 10 (Neural Networks)
# ============================================================

from typing import Dict, List
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Suppress TensorFlow logging warnings for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NeuralDiagnosticModel:
    """
    Deep Neural Network for medical diagnosis.
    Architecture: Input → Dense → BN → Dropout → Output
    """

    SYMPTOM_FEATURES = [
        'fever', 'cough', 'fatigue', 'difficulty_breathing',
        'muscle_ache', 'headache', 'sore_throat', 'nausea',
        'chills', 'vomiting', 'diarrhea', 'rash', 'loss_of_smell',
        'chest_pain', 'joint_pain', 'sweating', 'abdominal_pain', 'runny_nose'
    ]

    DISEASE_LABELS = [
        'flu', 'covid19', 'dengue', 'cardiac_event',
        'diabetes', 'common_cold', 'tuberculosis', 'meningitis'
    ]

    def __init__(self):
        self.model      = None
        self.history    = None
        self.is_trained = False
        self.label_encoder = LabelEncoder()
        
        # Paths for saving the compiled model and the label mapping
        self.model_path = 'data/dnn_model.h5'
        self.classes_path = 'data/dnn_classes.npy'
        
        self._build_model()

    def _build_model(self, n_outputs=None):
        """Build deep MLP architecture"""
        n_inputs  = len(self.SYMPTOM_FEATURES)
        if n_outputs is None:
            n_outputs = len(self.DISEASE_LABELS)

        self.model = models.Sequential([
            layers.Input(shape=(n_inputs,)),

            # Block 1: 32 nodes, ReLU activation with dropout
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),

            # Block 2: 16 nodes, ReLU activation
            layers.Dense(16, activation='relu'),

            # Output: Softmax activation for probability distribution
            layers.Dense(n_outputs, activation='softmax')
        ], name='MedicalDNN')

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def load_trained_model(self):
        """Loads the pre-trained Keras model and label mappings."""
        if os.path.exists(self.model_path) and os.path.exists(self.classes_path):
            self.model = load_model(self.model_path)
            self.label_encoder.classes_ = np.load(self.classes_path, allow_pickle=True)
            self.is_trained = True
            print(f"Models loaded from disk successfully.")
            return True
        else:
            print("No saved Neural Network found. You must call train_model() first.")
            return False

    def save_trained_model(self):
        """Save the trained model and label encoder classes."""
        os.makedirs('data', exist_ok=True)
        self.model.save(self.model_path)
        np.save(self.classes_path, self.label_encoder.classes_)
        print(f"Model saved to {self.model_path}")

    def train_model(self, csv_path="data/final_training_data.csv", epochs: int = 50):
        """Loads data from CSV, encodes labels, and trains the Deep Neural Network."""
        print(f"Loading data from {csv_path} for DNN training...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Could not find {csv_path}.")
            
        df = pd.read_csv(csv_path)
        
        # 1. Prepare Inputs (X)
        X = df[self.SYMPTOM_FEATURES].values
        
        # 2. Prepare Outputs (y)
        # Convert string labels into integer categories
        y_encoded = self.label_encoder.fit_transform(df['disease'])
        
        # Convert integers into One-Hot Encoded arrays
        y_categorical = to_categorical(y_encoded)
        
        num_classes = len(self.label_encoder.classes_)
        
        # 3. Split the Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_categorical, test_size=0.2, random_state=42
        )
        
        # 4. Build and Train
        print(f"Building MLP with 18 inputs and {num_classes} outputs...")
        self._build_model(num_classes)
        
        print(f"Training Neural Network for {epochs} epochs...")
        self.history = self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=32, 
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # 5. Evaluate Accuracy
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTraining Complete. Final Test Accuracy: {accuracy:.2%}")
        
        # 6. Save the model and the label encoder classes
        self.save_trained_model()
        self.is_trained = True
        return {'test_accuracy': accuracy}

    def _generate_data(self, n: int = 3000):
        """Generate synthetic training data"""
        np.random.seed(42)

        profiles = {
            'flu':           {'fever':0.90,'cough':0.85,'fatigue':0.88,
                              'headache':0.70,'body_aches':0.80},
            'covid19':       {'fever':0.88,'cough':0.80,'fatigue':0.90,
                              'loss_of_smell':0.85,'headache':0.65},
            'dengue':        {'fever':0.98,'rash':0.75,'joint_pain':0.85,
                              'headache':0.90,'fatigue':0.80},
            'cardiac_event': {'chest_pain':0.92,'shortness_of_breath':0.88,
                              'sweating':0.75,'fatigue':0.70},
            'diabetes':      {'fatigue':0.82,'frequent_urination':0.95,
                              'excessive_thirst':0.92,'blurred_vision':0.70},
            'common_cold':   {'cough':0.90,'fever':0.50,'headache':0.60,
                              'fatigue':0.55},
            'tuberculosis':  {'cough':0.95,'weight_loss':0.85,'night_sweats':0.80,
                              'fatigue':0.88,'fever':0.70},
            'meningitis':    {'headache':0.95,'stiff_neck':0.90,'fever':0.92,
                              'light_sensitivity':0.85},
        }

        X_list, y_list = [], []
        n_per = n // len(profiles)

        for label_idx, (disease, probs) in enumerate(profiles.items()):
            for _ in range(n_per):
                row = np.array([
                    1 if (np.random.random() <
                          probs.get(feat, 0.03)) else 0
                    for feat in self.SYMPTOM_FEATURES
                ], dtype=np.float32)
                X_list.append(row)
                y_list.append(label_idx)

        X = np.array(X_list)
        y = np.array(y_list)
        idx = np.random.permutation(len(X))
        return X[idx], y[idx]

    def train(self, csv_path="data/final_training_data.csv", use_synthetic=False, epochs: int = 50, verbose: int = 1) -> Dict:
        """Train the neural network. Attempts to load from CSV first, falls back to synthetic data."""
        # Try to load pre-trained models from disk first
        if self.load_trained_model():
            if verbose:
                print(f"Using pre-trained models from disk.")
            return {}
        
        # Try to train from CSV file if it exists
        if os.path.exists(csv_path) and not use_synthetic:
            try:
                return self.train_model(csv_path, epochs)
            except Exception as e:
                print(f"Warning: Could not train from CSV ({e}). Falling back to synthetic data.")
        
        # Fall back to synthetic data generation
        X, y = self._generate_data(3000)
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        cb_list = [
            callbacks.EarlyStopping(
                monitor='val_accuracy', patience=10,
                restore_best_weights=True),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5,
                patience=5, min_lr=1e-6)
        ]

        print("=" * 55)
        print("  Neural Network — Medical Diagnosis Training (Synthetic Data)")
        print(f"  Architecture: {len(self.SYMPTOM_FEATURES)} → "
              f"32 → 16 → {len(self.DISEASE_LABELS)}")
        print("=" * 55)
        self.model.summary()

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=64,
            callbacks=cb_list, verbose=verbose
        )

        val_acc = max(self.history.history['val_accuracy'])
        self.is_trained = True
        print(f"\n✅ Best Validation Accuracy: {val_acc:.4f}")
        return {'val_accuracy': val_acc}

    def predict(self, symptoms: List[str]) -> Dict:
        """Predict from symptom list"""
        if not self.is_trained:
            if not self.load_trained_model():
                self.train(verbose=0)

        features = np.array([
            [1.0 if feat in [s.lower().replace(' ','_')
                             for s in symptoms]
             else 0.0
             for feat in self.SYMPTOM_FEATURES]
        ], dtype=np.float32)

        proba     = self.model.predict(features, verbose=0)[0]
        pred_idx  = np.argmax(proba)
        diagnosis = self.DISEASE_LABELS[pred_idx]

        return {
            'diagnosis':  diagnosis,
            'confidence': round(float(proba[pred_idx]), 4),
            'all_probs':  dict(zip(self.DISEASE_LABELS,
                                   proba.round(4).tolist()))
        }

    def analyze(self, percept) -> Dict:
        """Module interface for the agent"""
        result = self.predict(percept.symptoms)
        result['summary'] = (f"DNN: {result['diagnosis']} "
                             f"({result['confidence']:.2%})")
        return result

    def plot_training(self):
        """Plot training history"""
        if not self.history:
            print("Train model first!")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        metrics = [('accuracy', 'val_accuracy', 'Accuracy'),
                   ('loss',     'val_loss',     'Loss')]
        colors  = [('#3498db','#e74c3c'), ('#2ecc71','#e67e22')]

        for ax, (train_m, val_m, title), (tc, vc) in zip(
                axes, metrics, colors):
            ax.plot(self.history.history[train_m],
                    color=tc, linewidth=2, label='Train')
            ax.plot(self.history.history[val_m],
                    color=vc, linewidth=2,
                    linestyle='--', label='Validation')
            ax.set_title(f"Model {title}",
                         fontsize=13, fontweight='bold')
            ax.set_xlabel("Epoch")
            ax.set_ylabel(title)
            ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle("Neural Network Training Curves",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("nn_training.png", dpi=150)
        plt.show()


if __name__ == "__main__":
    nn = NeuralDiagnosticModel()
    
    # Try to train from CSV file, with fallback to synthetic data
    print("Training from CSV file (data/final_training_data.csv)...\n")
    try:
        nn.train_model("data/final_training_data.csv", epochs=50)
    except Exception as e:
        print(f"CSV training failed ({e}). Using synthetic data instead.\n")
        nn.train(use_synthetic=True, verbose=1)
    
    # Test prediction
    result = nn.predict(['fever', 'cough', 'fatigue', 'loss_of_smell'])
    print(f"\nDiagnosis : {result['diagnosis']}")
    print(f"Confidence: {result['confidence']:.2%}")

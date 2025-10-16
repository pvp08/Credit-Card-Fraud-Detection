import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("fraud_detection.log"),
        logging.StreamHandler()
    ]
)

sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)


class FraudDetector:
    def __init__(self, data_path='data/creditcard.csv'):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}

    def load_data(self):
        logging.info("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        logging.info(f"Data loaded successfully. Shape: {self.df.shape}")
        return self

    def prepare_data(self, test_size=0.25, random_state=42):
        logging.info("Preparing data and applying SMOTE...")

        X = self.df.drop('Class', axis=1)
        y = self.df['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        logging.info(f"After SMOTE: {X_train_res.shape}, Fraud ratio: {y_train_res.mean():.4f}")

        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(X_train_res)
        self.X_test = scaler.transform(X_test)
        self.y_train = y_train_res
        self.y_test = y_test

        return self

    def train_models(self):
        logging.info("Training models...")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
            "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
            "XGBoost": XGBClassifier(scale_pos_weight=1, n_estimators=200, learning_rate=0.05, max_depth=5,
                                     random_state=42, n_jobs=-1)
        }

        for name, model in models.items():
            logging.info(f"Training {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            self._evaluate_model(model, name)

        return self

    def _evaluate_model(self, model, model_name):
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'f1': f1_score(self.y_test, y_pred),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba)
        }

        self.results[model_name] = metrics
        logging.info(f"{model_name} — Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")

    def compare_models(self):
        logging.info("Comparing models...")
        df = pd.DataFrame(self.results).T
        df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
        plt.title("Model Comparison")
        plt.ylabel("Score")
        plt.tight_layout()
        os.makedirs('reports/plots', exist_ok=True)
        plt.savefig('reports/plots/model_comparison.png', dpi=300)
        return df

    def save_models(self, path='models/'):
        os.makedirs(path, exist_ok=True)
        for name, model in self.models.items():
            file_path = f"{path}{name.replace(' ', '_').lower()}.pkl"
            joblib.dump(model, file_path)
            logging.info(f"Saved {name} to {file_path}")

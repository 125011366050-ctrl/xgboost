import os
import joblib
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any
from config import Config

logger = logging.getLogger(__name__)

class StackingPredictor:
    """Standalone XGBoost Predictor"""
    
    def __init__(self, config=Config):
        self.config = config
        self.models = []
        self.scaler = None
        self.horizons = [30, 60, 120]
        self._load_models()
    
    def _load_models(self):
        """Load standalone XGBoost models and scaler"""
        try:
            models_path = self.config.XGB_MODELS_PATH
            if not os.path.exists(models_path):
                raise FileNotFoundError(f"Models not found: {models_path}")
            
            self.models = joblib.load(models_path)
            print(f"Loaded {len(self.models)} XGBoost models")
            
            scaler_path = self.config.SCALER_PATH
            if not os.path.exists(scaler_path):
                raise FileNotFoundError(f"Scaler not found: {scaler_path}")
            
            self.scaler = joblib.load(scaler_path)
            print("Loaded glucose scaler")
            
        except Exception as e:
            print(f"Failed to load models: {e}")
            raise
    
    def _prepare_features(self, entries: pd.DataFrame) -> np.ndarray:
        """Convert input to model features (18 features × 36 timesteps)"""
        
        # Ensure we have 36 timesteps
        if len(entries) < self.config.WINDOW_SIZE:
            last_row = entries.iloc[-1]
            pad_rows = self.config.WINDOW_SIZE - len(entries)
            padded = pd.DataFrame([last_row] * pad_rows, columns=entries.columns)
            entries = pd.concat([padded, entries], ignore_index=True)
        elif len(entries) > self.config.WINDOW_SIZE:
            entries = entries.iloc[-self.config.WINDOW_SIZE:]
        
        features = []
        
        for idx, row in entries.iterrows():
            glucose = float(row.get('Glucose', 0))
            
            feat = [
                glucose,
                float(row.get('HR', 75)),
                float(row.get('Carbs', 0)),
                float(row.get('Protein', 0)),
                float(row.get('Fat', 0)),
                float(row.get('Fiber', 0)),
                float(row.get('Calories', 0)),
                float(row.get('Meal_Flag', 0)),
            ]
            
            if idx >= 6:
                window = entries.iloc[max(0, idx-6):idx+1]['Glucose']
                feat.extend([
                    float(window.mean()),
                    float(window.std()),
                    float(window.min()),
                    float(window.max()),
                    float((glucose - window.iloc[0]) / 6),
                ])
            else:
                feat.extend([glucose, 0, glucose, glucose, 0])
            
            if idx >= 3:
                window = entries.iloc[max(0, idx-3):idx+1]['Glucose']
                feat.extend([
                    float(window.mean()),
                    float((glucose - window.iloc[0]) / 3),
                ])
            else:
                feat.extend([glucose, 0])
            
            feat.extend([
                1 if glucose < 70 else 0,
                1 if glucose > 180 else 0,
                1 if 70 <= glucose <= 180 else 0,
            ])
            
            if len(feat) < 18:
                feat.extend([0] * (18 - len(feat)))
            elif len(feat) > 18:
                feat = feat[:18]
            
            features.append(feat)
        
        features = np.array(features)
        glucose_col = features[:, 0].reshape(-1, 1)
        glucose_scaled = self.scaler.transform(glucose_col).flatten()
        features[:, 0] = glucose_scaled
        
        return features.flatten()
    
    def predict(self, entries: pd.DataFrame) -> Dict[str, float]:
        """Predict glucose for all horizons (30, 60, 120 min)"""
        if entries is None or len(entries) == 0:
            raise ValueError("At least one CGM entry is required")
        
        features = self._prepare_features(entries)
        X = features.reshape(1, -1)
        
        predictions = {}
        for idx, horizon in enumerate(self.horizons):
            if idx < len(self.models):
                pred_scaled = self.models[idx].predict(X)[0]
                pred = self.scaler.inverse_transform([[pred_scaled]])[0][0]
                predictions[f"{horizon}min"] = float(pred)
        
        return predictions

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_type": "XGBoost Standalone",
            "num_models": len(self.models),
            "horizons": self.horizons,
            "feature_shape": (self.config.WINDOW_SIZE, self.config.N_FEATURES),
            "total_features": self.config.WINDOW_SIZE * self.config.N_FEATURES
        }

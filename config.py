import os

class Config:
    """CDSS Configuration for Standalone XGBoost"""

    # Food database (directly in main directory)
    EXCEL_PATH = "Indian_Foods_GI_GL_Database (1).xlsx"

    # XGBoost Models (directly in main directory)
    XGB_MODELS_PATH = "xgb_models.pkl"
    SCALER_PATH = "glucose_scaler.pkl"

    # Feature dimensions
    WINDOW_SIZE = 36
    N_FEATURES = 18

    # Model behavior
    DEFAULT_SENSITIVITY = 1.0
    DEFAULT_CARBS_LIMIT = 30

    MIN_SENSITIVITY = 0.1
    MAX_SENSITIVITY = 3.0

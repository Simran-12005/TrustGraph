"""
Dynamic Configuration Module for AML Rule Engine Thresholds.
"""

from pydantic import BaseModel


class AMLConfigModel(BaseModel):
    ctr_threshold: float = 10000.0
    structuring_window_hours: int = 24
    velocity_threshold_count: int = 5
    rapid_cashout_mins: int = 180
    medium_risk_threshold: float = 0.35
    high_risk_threshold: float = 0.65


class AMLConfigSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AMLConfigSingleton, cls).__new__(cls)
            cls._instance.config = AMLConfigModel()
        return cls._instance

    def update(self, new_settings: dict):
        current_data = self.config.model_dump()
        current_data.update(new_settings)
        self.config = AMLConfigModel(**current_data)

    def get(self) -> AMLConfigModel:
        return self.config


# Explicit export instantiated singleton
aml_config = AMLConfigSingleton()
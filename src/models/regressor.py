import xgboost as xgb


def build_regressor() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(
        n_estimators=100,
        random_state=42,
    )

    return model

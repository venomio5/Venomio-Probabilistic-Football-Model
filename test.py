import core
import ast
from sklearn.linear_model import Ridge
import scipy.sparse as sp
import numpy as np
import xgboost as xgb
import pandas as pd

def train_refined_sq_model() -> tuple[xgb.Booster, list[str]]:
    sql = """
        SELECT
            total_plsqa,
            shooter_sq,
            assister_sq,
            CASE WHEN match_state < 0 THEN 'Trailing'
                    WHEN match_state = 0 THEN 'Level'
                    ELSE 'Leading' END AS match_state,
            CASE WHEN player_dif < 0 THEN 'Neg'
                    WHEN player_dif = 0 THEN 'Neu'
                    ELSE 'Pos' END      AS player_dif,
            xg
        FROM shots_data
        WHERE total_plsqa IS NOT NULL
    """
    df = core.DB.select(sql)
    print(df)

    cat_cols = ['match_state', 'player_dif']
    num_cols = ['total_plsqa', 'shooter_sq', 'assister_sq']
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors='coerce')
    for c in cat_cols:
        df[c] = df[c].astype(str)

    X_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, dummy_na=True)
    X     = pd.concat([df[num_cols], X_cat], axis=1).astype(float)
    y     = df['xg'].astype(float)

    dtrain = xgb.DMatrix(X, label=y)
    params = dict(objective='reg:squarederror', eval_metric='rmse',
                    tree_method='hist', max_depth=6, eta=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=2)
    booster = xgb.train(params, dtrain, num_boost_round=400)
    return booster, X.columns.tolist()

booster, feature_names = train_refined_sq_model()

def predict_custom_input(feature_values: dict):
    """
    Given a dictionary of feature values, build a DataFrame,
    align it to the trained feature space, and make prediction.
    """
    # Create default zero row
    input_df = pd.DataFrame([0.0] * len(feature_names), index=feature_names).T

    # Assign provided values
    for key, val in feature_values.items():
        if key in input_df.columns:
            input_df.at[0, key] = val
        else:
            print(f"Warning: Feature '{key}' not recognized in trained model.")

    # Create DMatrix and predict
    dtest = xgb.DMatrix(input_df)
    prediction = booster.predict(dtest)
    return prediction[0]

# === Example Usage ===
if __name__ == "__main__":
    # Modify these values to probe model behavior
    input_features = {
        'total_plsqa': 0.1,
        'shooter_sq': 0.75,
        'assister_sq': 0.55,
        'match_state_Trailing': 1.0,
        'player_dif_Pos': 1.0
    }

    result = predict_custom_input(input_features)
    print(f"Predicted xG: {result:.4f}")
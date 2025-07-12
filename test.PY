# import core
# core.AutoLineups(3102)

import core
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


ras_booster, ras_cr_columns = core._load_model(f'psxg')

importance = ras_booster.get_score(importance_type='gain') 
keys = list(importance.keys())
values = [importance[k] for k in keys]

plt.figure(figsize=(10, 6))
plt.barh(keys, values)
plt.xlabel('Importance')
plt.title('Feature Importance (gain)')
plt.gca().invert_yaxis()
plt.show()

def predict_context_ras(self, booster, feature_columns, new_match, *, raw=False):
    categorical_cols = ['match_state', 'match_segment', 'player_dif', 'match_time']
    bool_cols = ['team_is_home', 'is_raining']
    num_cols = ['team_elevation_dif', 'opp_elevation_dif', 'team_travel', 'opp_travel', 'team_rest_days', 'opp_rest_days', 'temperature_c']
    
    base_margin = np.log(new_match.pop('total_ras').clip(lower=1e-6))

    for col in categorical_cols:
        new_match[col] = new_match[col].astype(str).str.lower()
    new_match[bool_cols] = new_match[bool_cols].astype(int)

    new_cat = pd.get_dummies(new_match[categorical_cols], prefix=categorical_cols)
    new_cat = new_cat.reindex(columns=[col for col in feature_columns if any(col.startswith(prefix + "_") for prefix in categorical_cols)], fill_value=0)

    new_X = pd.concat([
        new_match[num_cols + bool_cols].reset_index(drop=True),
        new_cat.reset_index(drop=True)
    ], axis=1)

    new_X = new_X.reindex(columns=feature_columns, fill_value=0)

    dmatrix = xgb.DMatrix(new_X, base_margin=base_margin)
    prediction = booster.predict(dmatrix, output_margin=raw)
    return prediction[0]
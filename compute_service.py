import os
from fastapi import FastAPI, Header, HTTPException
import uvicorn
import core
from datetime import datetime, timedelta
import pandas as pd
import math
import numpy as np

# SQL
def get_all_matches():
    fixtures_query = "SELECT * FROM schedule_data"
    matches = core.DB.select(fixtures_query)

    matches["datetime"] = matches.apply(
        lambda row: datetime.combine(row["date"], datetime.min.time()) + row["local_time"],
        axis=1
    )
    matches["league_name"] = matches["league_id"].apply(lambda lid: core.get_league_name_by_id(lid))
    matches["home_team"] = matches["home_team_id"].apply(lambda tid: core.get_team_name_by_id(tid))
    matches["away_team"] = matches["away_team_id"].apply(lambda tid: core.get_team_name_by_id(tid))

    return matches

def load_simulation_df(schedule_id: int):
    shots_df = core.DB.select("SELECT * FROM simulation_data WHERE schedule_id = %s", (schedule_id,))

    meta_sql = """
        SELECT 
            date,
            home_team_id,
            away_team_id,
            current_home_goals,
            current_away_goals,
            current_period_start_timestamp,
            period,
            period_injury_time
        FROM schedule_data
        WHERE schedule_id = %s
    """

    match_df = core.DB.select(meta_sql, (schedule_id,))

    if match_df.empty:
        raise ValueError(f"schedule_id {schedule_id} not found in schedule_data.")

    row = match_df.iloc[0]

    metadata = {
        "datetime": pd.to_datetime(row["date"]),
        "home_id": int(row["home_team_id"]),
        "away_id": int(row["away_team_id"]),
        "home_goals": int(row["current_home_goals"]),
        "away_goals": int(row["current_away_goals"]),
        "period_start": row["current_period_start_timestamp"],
        "period": row["period"],
        "injury_time": int(row["period_injury_time"]) if row["period_injury_time"] and not math.isnan(row["period_injury_time"]) else 0
    }

    return shots_df, metadata

def get_aggregated_goals(shots_df, home_team_id, start_minute, start_home_goals, start_away_goals):
    if shots_df is None or shots_df.empty:
        return pd.DataFrame(columns=['sim_id', 'minute', 'home_goals', 'away_goals'])
    
    df = shots_df.copy()
    df = df[df['minute'] >= start_minute]
    
    df = df.sort_values(['sim_id', 'minute']).reset_index(drop=True)
    
    df['squad']   = pd.to_numeric(df['squad'],   errors='coerce').astype('Int64')
    df['outcome'] = pd.to_numeric(df['outcome'], errors='coerce').fillna(0).astype(int)
    
    df['is_home']   = df['squad'] == int(home_team_id)
    df['home_goal'] = ((df['outcome'] == 1) &  df['is_home']).astype(int)
    df['away_goal'] = ((df['outcome'] == 1) & ~df['is_home']).astype(int)
    
    df['home_goal_cum'] = df.groupby('sim_id')['home_goal'].cumsum() + start_home_goals
    df['away_goal_cum'] = df.groupby('sim_id')['away_goal'].cumsum() + start_away_goals
    
    agg = (
        df.groupby(['sim_id', 'minute'])
        .agg(home_goals=('home_goal_cum', 'last'),
            away_goals=('away_goal_cum', 'last'))
        .reset_index()
    )
    
    max_minute = 90
    full_index = pd.MultiIndex.from_product(
        [agg['sim_id'].unique(), range(max_minute + 1)],
        names=['sim_id', 'minute']
    )
    
    agg = (
        agg.set_index(['sim_id', 'minute'])
        .reindex(full_index)
        .groupby(level=0)
        .ffill()
        .fillna({'home_goals': start_home_goals, 'away_goals': start_away_goals})
        .reset_index()
    )
    return agg

def get_odds(schedule_id: int, market_key: str) -> dict:
    shots_df, metadata = load_simulation_df(schedule_id)
    if shots_df is None or shots_df.empty or metadata.get("home_id") is None or metadata.get("away_id") is None:
        return {}

    kickoff = metadata["datetime"]
    current_period_start_timestamp = metadata["period_start"]
    period = metadata["period"]
    injury_time = metadata["injury_time"]

    now = datetime.now()

    if not (timedelta(hours=0) <= now - kickoff <= timedelta(hours=2.1)):
        current_minute = 0

    current_period_start = datetime.fromtimestamp(current_period_start_timestamp)
    elapsed_minutes = int((now - current_period_start).total_seconds() // 60)

    base_minute = 0
    if period == "period2":
        base_minute = 45

    current_minute = base_minute + min(elapsed_minutes, 45)

    if injury_time is not None:
        current_minute = current_minute - injury_time

    if period == "period1":
        extra_time = 2
        if current_minute >= 30:
            estimated = (extra_time / 15) * (current_minute - 30)
            current_minute = int(current_minute - estimated)
    elif period == "period2":
        extra_time = 5
        if current_minute >= 60:
            estimated = (extra_time / 15) * (current_minute - 60)
            current_minute = int(current_minute - estimated)

    current_minute = max(current_minute, 0)

    agg = get_aggregated_goals(shots_df, metadata.get("home_id"), current_minute, metadata.get("home_goals"), metadata.get("away_goals"))
    filtered_data = agg[
        (agg["minute"] == current_minute) &
        (agg["home_goals"] == metadata.get("home_goals")) &
        (agg["away_goals"] == metadata.get("away_goals"))
    ]
    relevant_sim_ids = filtered_data["sim_id"].unique()
    if len(relevant_sim_ids) == 0:
        return {}
    final_minute = agg["minute"].max()
    final_df = agg[(agg["minute"] == final_minute) & (agg["sim_id"].isin(relevant_sim_ids))]
    total = len(final_df)
    if total == 0:
        return {}

    if market_key == "ft_result":
        home_wins = len(final_df[final_df["home_goals"] > final_df["away_goals"]])
        away_wins = len(final_df[final_df["home_goals"] < final_df["away_goals"]])
        draws     = total - home_wins - away_wins
        return {
            "home": round(1 / (home_wins / total), 3) if home_wins else 0,
            "away": round(1 / (away_wins / total), 3) if away_wins else 0,
            "draw": round(1 / (draws     / total), 3) if draws     else 0,
        }

    if market_key == "asian_handicap":
        result = {}
        for hcap in ["2.5", "1.5"]:
            val = float(hcap)
            home_plus = len(final_df[final_df["home_goals"] + val > final_df["away_goals"]])
            away_minus = total - home_plus

            home_minus = len(final_df[final_df["home_goals"] - val > final_df["away_goals"]])
            away_plus = total - home_minus

            result[f"+{hcap}"] = {
                "home": round(1 / (home_plus / total), 3) if home_plus else 0,
                "away": round(1 / (away_minus / total), 3) if away_minus else 0,
            }
            result[f"-{hcap}"] = {
                "home": round(1 / (home_minus / total), 3) if home_minus else 0,
                "away": round(1 / (away_plus / total), 3) if away_plus else 0,
            }
        return result

    if market_key == "total_goals":
        result = {}
        for t in ["0.5", "1.5", "2.5", "3.5", "4.5", "5.5"]:
            threshold = float(t)
            under = len(final_df[final_df["home_goals"] + final_df["away_goals"] < threshold])
            over  = len(final_df[final_df["home_goals"] + final_df["away_goals"] > threshold])
            result[t] = {
                "over":  round(1 / (over  / total), 3) if over  else 0,
                "under": round(1 / (under / total), 3) if under else 0,
            }
        return result

    if market_key == "correct_score":
        result = {}
        total_final_data = total
        score_counts = {}
        for _, row in final_df.iterrows():
            key = (row["home_goals"], row["away_goals"])
            score_counts[key] = score_counts.get(key, 0) + 1
        for score in [
            "0-0", "0-1", "0-2", "0-3",
            "1-0", "1-1", "1-2", "1-3",
            "2-0", "2-1", "2-2", "2-3",
            "3-0", "3-1", "3-2", "3-3"
        ]:
            home_goals, away_goals = map(int, score.split("-"))
            count = score_counts.get((home_goals, away_goals), 0)
            odds_val = round(1 / (count / total_final_data), 3) if total_final_data != 0 and count != 0 else 10**30
            result[score] = odds_val
        any_other_home_win = sum(score_counts.get((h, a), 0) for h in range(4, 11) for a in range(0, 4))
        any_other_away_win = sum(score_counts.get((h, a), 0) for h in range(0, 4) for a in range(4, 11))
        any_other_draw = sum(score_counts.get((h, h), 0) for h in range(4, 11))
        aggregated_home_odds = round(1 / (any_other_home_win / total_final_data), 3) if total_final_data != 0 and any_other_home_win != 0 else 10**30
        aggregated_away_odds = round(1 / (any_other_away_win / total_final_data), 3) if total_final_data != 0 and any_other_away_win != 0 else 10**30
        aggregated_draw_odds = round(1 / (any_other_draw / total_final_data), 3) if total_final_data != 0 and any_other_draw != 0 else 10**30
        result["Local +4"] = aggregated_home_odds
        result["Visitante +4"] = aggregated_away_odds
        result["Empate +4"] = aggregated_draw_odds
        return result
    
    if market_key == "team_totals":
        result = {"home": {}, "away": {}}
        for t in ["0.5", "1.5", "2.5"]:
            threshold = float(t)

            over_home = len(final_df[final_df["home_goals"] > threshold])
            under_home = total - over_home
            result["home"][t] = {
                "over": round(1 / (over_home / total), 3) if over_home else 0,
                "under": round(1 / (under_home / total), 3) if under_home else 0,
            }

            over_away = len(final_df[final_df["away_goals"] > threshold])
            under_away = total - over_away
            result["away"][t] = {
                "over": round(1 / (over_away / total), 3) if over_away else 0,
                "under": round(1 / (under_away / total), 3) if under_away else 0,
            }

        return result
    
    if market_key == "btts":
        both_score = len(final_df[(final_df["home_goals"] > 0) & (final_df["away_goals"] > 0)])
        not_both_score = total - both_score
        return {
            "yes": round(1 / (both_score / total), 3) if both_score else 0,
            "no":  round(1 / (not_both_score / total), 3) if not_both_score else 0,
        }

    if market_key == "player_goals":
        query = """
            SELECT shooter,
                   COUNT(DISTINCT sim_id) AS sims_with_goal,
                   (SELECT COUNT(DISTINCT sim_id)
                      FROM simulation_data
                      WHERE schedule_id = %(schedule_id)s) AS total_sims,
                   COUNT(DISTINCT sim_id) * 1.0 /
                   (SELECT COUNT(DISTINCT sim_id)
                      FROM simulation_data
                      WHERE schedule_id = %(schedule_id)s) AS goal_pct
            FROM simulation_data
            WHERE schedule_id = %(schedule_id)s
              AND outcome = 1
            GROUP BY shooter
            ORDER BY goal_pct DESC
            LIMIT 15
        """
        params = {"schedule_id": schedule_id}
        rows = core.DB.select(query, params)
        odds = {}
        for _, row in rows.iterrows():
            pct = float(row["goal_pct"])
            odds_val = round(1 / pct, 3) if pct > 0 else 10**30
            odds[row["shooter"]] = odds_val
        return odds

    if market_key == "player_assists":
        query = """
            SELECT assister,
                   COUNT(DISTINCT sim_id) AS sims_with_assist,
                   (SELECT COUNT(DISTINCT sim_id)
                      FROM simulation_data
                      WHERE schedule_id = %(schedule_id)s) AS total_sims,
                   COUNT(DISTINCT sim_id) * 1.0 /
                   (SELECT COUNT(DISTINCT sim_id)
                      FROM simulation_data
                      WHERE schedule_id = %(schedule_id)s) AS assist_pct
            FROM simulation_data
            WHERE schedule_id = %(schedule_id)s
              AND outcome = 1
              AND assister IS NOT NULL
            GROUP BY assister
            ORDER BY assist_pct DESC
            LIMIT 10
        """
        params = {"schedule_id": schedule_id}
        rows = core.DB.select(query, params)
        odds = {}
        for _, row in rows.iterrows():
            pct = float(row["assist_pct"])
            odds_val = round(1 / pct, 3) if pct > 0 else 10**30
            odds[row["assister"]] = odds_val
        return odds

    if market_key == "player_shots":
        thresholds = {"0.5": 1, "1.5": 2, "2.5": 3, "3.5": 4}
        total_query = """
            SELECT COUNT(DISTINCT sim_id) AS total_sims
            FROM simulation_data
            WHERE schedule_id = %(schedule_id)s
        """
        total_result = core.DB.select(total_query, {"schedule_id": schedule_id})
        total_sims = total_result.iloc[0]["total_sims"] if not total_result.empty else 0
        odds = {}
        for th_str, req in thresholds.items():
            query = f"""
                SELECT shooter,
                       COUNT(DISTINCT sim_id) AS sims_with_shots
                FROM (
                    SELECT sim_id, shooter, COUNT(*) AS shot_count
                    FROM simulation_data
                    WHERE schedule_id = %(schedule_id)s
                    GROUP BY sim_id, shooter
                ) t
                WHERE shot_count >= {req}
                GROUP BY shooter
                ORDER BY sims_with_shots DESC
                LIMIT 5
            """
            params = {"schedule_id": schedule_id}
            rows = core.DB.select(query, params)
            sub_odds = {}
            for _, row in rows.iterrows():
                if total_sims > 0:
                    pct = float(row["sims_with_shots"]) / total_sims
                else:
                    pct = 0
                odds_val = round(1 / pct, 3) if pct > 0 else 10**30
                sub_odds[row["shooter"]] = odds_val
            odds[th_str] = sub_odds
        return odds

    return {}

def _df_to_jsonable(df: pd.DataFrame) -> list[dict]:
    return (
        df.replace({np.nan: None, np.inf: None, -np.inf: None})
          .to_dict(orient="records")
    )

API_KEY = os.getenv("COMPUTE_API_KEY", "")

app = FastAPI()

def _auth(x_api_key: str = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorised")

@app.get("/ping")
async def ping():
    return {"status": "ok"}


@app.get("/matches")
async def matches(x_api_key: str = Header(None)):
    _auth(x_api_key)
    df = get_all_matches()
    return _df_to_jsonable(df)


@app.get("/match/{schedule_id}")
async def match_detail(schedule_id: int, x_api_key: str = Header(None)):
    _auth(x_api_key)
    df = get_all_matches()
    row = df[df["schedule_id"] == schedule_id]
    if row.empty:
        raise HTTPException(status_code=404, detail="not found")
    return _df_to_jsonable(row)[0] 


@app.get("/odds/{schedule_id}/{market_key}")
async def odds(schedule_id: int, market_key: str, x_api_key: str = Header(None)):
    _auth(x_api_key)
    return get_odds(schedule_id, market_key)


if __name__ == "__main__":
    uvicorn.run("compute_service:app", host="0.0.0.0", port=8001)
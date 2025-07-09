import requests
from rapidfuzz import process
import core
import unicodedata
import json

def _clean(txt: str) -> str:
    return unicodedata.normalize("NFKD", txt).encode("ascii", "ignore").decode().lower()

def _match_player_id(api_name: str, squad_names: dict[str, str]) -> str | None:
    api_name = _clean(api_name)

    if api_name in squad_names:
        return squad_names[api_name]

    best = process.extractOne(api_name, squad_names.keys(), score_cutoff=70)
    return squad_names[best[0]] if best else None

def parse_incidents(
    incidents: list[dict],
    home_status: list[dict],
    away_status: list[dict],
    last_minute_checked: int = 0
) -> tuple[dict, list[dict], list[dict], int, int, int]:
    events = {
        "home": {"substitutions": [], "yellow_cards": [], "red_cards": []},
        "away": {"substitutions": [], "yellow_cards": [], "red_cards": []},
    }

    def _build_idx(players_status):
        return {
            _clean(p["player_id"].split("_")[0]): p["player_id"]
            for p in players_status
        }

    idx_home, idx_away = _build_idx(home_status), _build_idx(away_status)

    last_minute = last_minute_checked
    cnt = {
        "home": {"sub": 0, "yellow": 0, "red": 0},
        "away": {"sub": 0, "yellow": 0, "red": 0},
    }

    # ------------------------------------------------------------------------    
    for inc in incidents:
        minute     = inc.get("time", 0)
        if minute <= last_minute_checked:
            continue

        side       = "home" if inc.get("isHome") else "away"
        squad_idx  = idx_home if side == "home" else idx_away
        squad_stat = home_status if side == "home" else away_status

        inc_type   = inc.get("incidentType")
        last_minute = max(last_minute, minute)

        # substitutions -------------------------------------------------------
        if inc_type == "substitution":
            cnt[side]["sub"] += 1
            pid_in  = _match_player_id(inc["playerIn"]["name"],  squad_idx)
            pid_out = _match_player_id(inc["playerOut"]["name"], squad_idx)

            for pl in squad_stat:
                if pl["player_id"] == pid_in:
                    pl.update({"bench": False, "on_field": True})
                elif pl["player_id"] == pid_out:
                    pl.update({"bench": False, "on_field": False})

            events[side]["substitutions"].append(
                {"minute": minute, "in": pid_in, "out": pid_out}
            )

        # cards ----------------------------------------------------------------
        elif inc_type == "card":
            pid = _match_player_id(inc["player"]["name"], squad_idx)

            if inc["incidentClass"] == "yellow":
                cnt[side]["yellow"] += 1
                events[side]["yellow_cards"].append({"minute": minute, "player": pid})
                for pl in squad_stat:
                    if pl["player_id"] == pid:
                        pl["yellow_card"] = True

            elif inc["incidentClass"] == "red":
                cnt[side]["red"] += 1
                events[side]["red_cards"].append({"minute": minute, "player": pid})
                for pl in squad_stat:
                    if pl["player_id"] == pid:
                        pl["red_card"] = True

    # simulate flags ----------------------------------------------------------
    simulate_home = int(
        cnt["home"]["sub"] > 0 or cnt["home"]["red"] > 0 or cnt["home"]["yellow"] >= 2
    )
    simulate_away = int(
        cnt["away"]["sub"] > 0 or cnt["away"]["red"] > 0 or cnt["away"]["yellow"] >= 2
    )

    return (
        events,
        home_status,
        away_status,
        last_minute,
        simulate_home,
        simulate_away,
    )

sql_query = f"""
    SELECT 
        *
    FROM schedule_data
    WHERE schedule_id = '1336';
"""
result = core.DB.select(sql_query)
home_players_data = result['home_players_data'].iloc[0]
away_players_data = result['away_players_data'].iloc[0]
last_minute_checked = int(result['last_minute_checked'].iloc[0]) or 0

match_id = 10340773
headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/85.0.4183.121 Safari/537.36"
    ),
    "Accept": "application/json",
}
api_incidents_url = f"https://www.sofascore.com/api/v1/event/{match_id}/incidents"
iresponse = requests.get(api_incidents_url, headers=headers)

incidents_data = iresponse.json()["incidents"]

upd_home, upd_away, last_min, sim_home, sim_away = parse_incidents(
    incidents_data,
    json.loads(home_players_data),
    json.loads(away_players_data),
    last_minute_checked  
)

simulate = int(sim_home or sim_away)

core.DB.execute(
    """
    UPDATE schedule_data
    SET home_players_data = %s,
        away_players_data = %s,
        last_minute_checked = %s,
        simulate        = %s
    WHERE schedule_id    = %s;
    """,
    (
        json.dumps(upd_home),
        json.dumps(upd_away),
        last_min,
        simulate,
        self.schedule_id
    )
)
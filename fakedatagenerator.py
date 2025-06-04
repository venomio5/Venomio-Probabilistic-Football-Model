"""
Código para generar datos fake para el modelo y SQL.
Se definen equipos y jugadores conocidos.
Se generan estadísticas aleatorias ponderadas usando parámetros base y modificadores (como +10% si se juega en casa, –5% si se está de visitante).
Se generan INSERTs SQL para las tres tablas:
• match_info
• match_detail
• match_breakdown
"""

import random, json
from datetime import datetime, timedelta

random.seed(42)

N_PARTIDOS = 5   # número de partidos a generar

"""
Definición de equipos y jugadores
Cada equipo tendrá: id, nombre y lista de jugadores. Cada jugador se define con:
- nombre, posición (GK, DEF, MED, ATT)
- parámetros base (shots per minute, shooting_ability)
"""

teams = {
    1: {
        "name": "Real Madrid",
        "players": [
            {"id": "T1_P1", "name": "Alisson Becker", "pos": "GK", "base_spm": 0.01, "shooting": 0.05},
            {"id": "T1_P2", "name": "Virgil van Dijk", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T1_P3", "name": "Ronald Araújo", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T1_P4", "name": "Kevin De Bruyne", "pos": "MED", "base_spm": 0.04, "shooting": 0.65},
            {"id": "T1_P5", "name": "Pedri González", "pos": "MED", "base_spm": 0.04, "shooting": 0.60},
            {"id": "T1_P6", "name": "Victor Osimhen", "pos": "ATT", "base_spm": 0.06, "shooting": 0.80},
            {"id": "T1_P7", "name": "Lautaro Martínez", "pos": "ATT", "base_spm": 0.06, "shooting": 0.85}
        ]
    },
    2: {
        "name": "Manchester City",
        "players": [
            {"id": "T2_P1", "name": "Emiliano Martínez", "pos": "GK", "base_spm": 0.01, "shooting": 0.05},
            {"id": "T2_P2", "name": "William Saliba", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T2_P3", "name": "Mats Hummels", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T2_P4", "name": "Jude Bellingham", "pos": "MED", "base_spm": 0.04, "shooting": 0.65},
            {"id": "T2_P5", "name": "Nicolò Barella", "pos": "MED", "base_spm": 0.04, "shooting": 0.60},
            {"id": "T2_P6", "name": "Karim Benzema", "pos": "ATT", "base_spm": 0.06, "shooting": 0.85},
            {"id": "T2_P7", "name": "Cody Gakpo", "pos": "ATT", "base_spm": 0.06, "shooting": 0.80}
        ]
    },
    3: {
        "name": "Bayern Múnich",
        "players": [
            {"id": "T3_P1", "name": "Yassine Bounou", "pos": "GK", "base_spm": 0.01, "shooting": 0.05},
            {"id": "T3_P2", "name": "Kalidou Koulibaly", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T3_P3", "name": "José María Giménez", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T3_P4", "name": "Bruno Fernandes", "pos": "MED", "base_spm": 0.04, "shooting": 0.65},
            {"id": "T3_P5", "name": "Weston McKennie", "pos": "MED", "base_spm": 0.04, "shooting": 0.60},
            {"id": "T3_P6", "name": "Paulo Dybala", "pos": "ATT", "base_spm": 0.06, "shooting": 0.85},
            {"id": "T3_P7", "name": "João Félix", "pos": "ATT", "base_spm": 0.06, "shooting": 0.80}
        ]
    },
    4: {
        "name": "Paris Saint-Germain",
        "players": [
            {"id": "T4_P1", "name": "Jan Oblak", "pos": "GK", "base_spm": 0.01, "shooting": 0.05},
            {"id": "T4_P2", "name": "Marquinhos", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T4_P3", "name": "Rúben Dias", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T4_P4", "name": "Martin Ødegaard", "pos": "MED", "base_spm": 0.04, "shooting": 0.65},
            {"id": "T4_P5", "name": "Toni Kroos", "pos": "MED", "base_spm": 0.04, "shooting": 0.60},
            {"id": "T4_P6", "name": "Darwin Núñez", "pos": "ATT", "base_spm": 0.06, "shooting": 0.85},
            {"id": "T4_P7", "name": "Olivier Giroud", "pos": "ATT", "base_spm": 0.06, "shooting": 0.80}
        ]
    },
    5: {
        "name": "Juventus",
        "players": [
            {"id": "T5_P1", "name": "Mike Maignan", "pos": "GK", "base_spm": 0.01, "shooting": 0.05},
            {"id": "T5_P2", "name": "Sergio Ramos", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T5_P3", "name": "Raphaël Varane", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T5_P4", "name": "Luka Modrić", "pos": "MED", "base_spm": 0.04, "shooting": 0.65},
            {"id": "T5_P5", "name": "Declan Rice", "pos": "MED", "base_spm": 0.04, "shooting": 0.60},
            {"id": "T5_P6", "name": "Erling Haaland", "pos": "ATT", "base_spm": 0.06, "shooting": 0.90},
            {"id": "T5_P7", "name": "Ansu Fati", "pos": "ATT", "base_spm": 0.06, "shooting": 0.80}
        ]
    },
    6: {
        "name": "Chelsea",
        "players": [
            {"id": "T6_P1", "name": "Thibaut Courtois", "pos": "GK", "base_spm": 0.01, "shooting": 0.05},
            {"id": "T6_P2", "name": "Aymeric Laporte", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T6_P3", "name": "Thiago Silva", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T6_P4", "name": "Gavi Páez", "pos": "MED", "base_spm": 0.04, "shooting": 0.65},
            {"id": "T6_P5", "name": "Adrien Rabiot", "pos": "MED", "base_spm": 0.04, "shooting": 0.60},
            {"id": "T6_P6", "name": "Harry Kane", "pos": "ATT", "base_spm": 0.06, "shooting": 0.90},
            {"id": "T6_P7", "name": "Rafael Leão", "pos": "ATT", "base_spm": 0.06, "shooting": 0.80}
        ]
    },
    7: {
        "name": "Barcelona",
        "players": [
            {"id": "T7_P1", "name": "Marc-André ter Stegen", "pos": "GK", "base_spm": 0.01, "shooting": 0.05},
            {"id": "T7_P2", "name": "Milan Škriniar", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T7_P3", "name": "Eder Militão", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T7_P4", "name": "Federico Valverde", "pos": "MED", "base_spm": 0.04, "shooting": 0.65},
            {"id": "T7_P5", "name": "Youri Tielemans", "pos": "MED", "base_spm": 0.04, "shooting": 0.60},
            {"id": "T7_P6", "name": "Gabriel Jesus", "pos": "ATT", "base_spm": 0.06, "shooting": 0.90},
            {"id": "T7_P7", "name": "Dusan Vlahović", "pos": "ATT", "base_spm": 0.06, "shooting": 0.80}
        ]
    },
    8: {
        "name": "Liverpool",
        "players": [
            {"id": "T8_P1", "name": "David de Gea", "pos": "GK", "base_spm": 0.01, "shooting": 0.05},
            {"id": "T8_P2", "name": "Benjamin Pavard", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T8_P3", "name": "Jules Koundé", "pos": "DEF", "base_spm": 0.02, "shooting": 0.10},
            {"id": "T8_P4", "name": "Sandro Tonali", "pos": "MED", "base_spm": 0.04, "shooting": 0.65},
            {"id": "T8_P5", "name": "Eduardo Camavinga", "pos": "MED", "base_spm": 0.04, "shooting": 0.60},
            {"id": "T8_P6", "name": "Antoine Griezmann", "pos": "ATT", "base_spm": 0.06, "shooting": 0.90},
            {"id": "T8_P7", "name": "Roberto Firmino", "pos": "ATT", "base_spm": 0.06, "shooting": 0.80}
        ]
    }
}

team_elevations = {
    1: 667,
    2: 52,
    3: 519,
    4: 35,
    5: 239,
    6: 11,
    7: 12,
    8: 70
}

average_elevation = sum(team_elevations.values()) / len(team_elevations)

def get_player(team_id, player_id):
    for p in teams[team_id]["players"]:
        if p["id"] == player_id:
            return p
    return None

#Funciones de ayuda para generar datos aleatorios
def random_date(start, end):
    """Genera una fecha aleatoria entre start y end (objetos datetime)"""
    delta = end - start
    random_seconds = random.randrange(delta.days * 24 * 3600)
    return start + timedelta(seconds=random_seconds)

def simula_match_info(match_id, home_team_id, away_team_id):
    """Genera datos fake para la tabla match_info"""
    # Fecha de partido aleatoria en el último año
    match_date = random_date(datetime.now()-timedelta(days=365), datetime.now())
    # Liga y árbitro aleatorios
    match_league_id = 1
    match_referee_id = random.randint(1,10)
    # Totales de cartas y faltas
    match_total_fouls = random.randint(10,30)
    match_yellow_cards = random.randint(0, 5)
    match_red_cards = random.randint(0, 2)
    minutes_played = 90
    # Elevación y travel aleatorios (con pequeñas diferencias)
    home_elev = round((team_elevations[home_team_id] / 1000) - (average_elevation / 1000), 3)
    away_elev = round((team_elevations[away_team_id] / 1000) - (average_elevation / 1000), 3)
    home_travel = 0
    away_travel = round(random.uniform(0, 100), 1)
    home_rest_days = random.randint(1,7)
    away_rest_days = random.randint(1,7)
    home_importance = random.choice([0, 1])
    away_importance = random.choice([0, 1])
    temperature_c = round(random.uniform(10,40), 1)
    is_raining = random.choice([0, 1])
    match_time = random.choice(["aft", "evening", "night"])

    info = {
        "match_id": match_id,
        "match_home_team_id": home_team_id,
        "match_away_team_id": away_team_id,
        "match_date": match_date.strftime("%Y-%m-%d %H:%M:%S"),
        "match_league_id": match_league_id,
        "match_referee_id": match_referee_id,
        "match_total_fouls": match_total_fouls,
        "match_yellow_cards": match_yellow_cards,
        "match_red_cards": match_red_cards,
        "minutes_played": minutes_played,
        "home_elevation_dif": home_elev,
        "away_elevation_dif": away_elev,
        "home_travel": home_travel,
        "away_travel": away_travel,
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "home_importance": home_importance,
        "away_importance": away_importance,
        "temperature_c": temperature_c,
        "is_raining": is_raining,
        "match_time": match_time
    }
    return info

def simula_match_detail(match_id, home_team_id, away_team_id):
    """Genera datos fake para la tabla match_detail.
    Se incluye la lista de jugadores a modo de JSON y se generan algunos contadores (headers, footers, xG, etc) aleatorios.
    También se incluyen las variables categóricas (match_state, match_segment, player_dif).
    """

    # Generar datos de tiros (algunos números base, con efecto home/away)
    # Ejemplo: headers y footers totales, xG por vía de headers y footers.
    # Además se aplica un ajuste: si el equipo en cuestión es "casa" se aumenta un 10% en los tiros.
    home_ajuste = 1.10
    away_ajuste = 0.95

    teamA_headers = int(random.randint(0,5) * home_ajuste)
    teamA_footers = int(random.randint(3,8) * home_ajuste)
    teamA_hxg = round(random.uniform(0.0, 1.5)* home_ajuste, 2)
    teamA_fxg = round(random.uniform(0.5, 2.0)* home_ajuste, 2)

    teamB_headers = int(random.randint(0,5) * away_ajuste)
    teamB_footers = int(random.randint(3,8) * away_ajuste)
    teamB_hxg = round(random.uniform(0.0, 1.5)* away_ajuste, 2)
    teamB_fxg = round(random.uniform(0.5, 2.0)* away_ajuste, 2)

    match_state = random.choice(['-1.5','-1','0','1','1.5'])
    match_segment = random.choice(['1','2','3','4','5','6'])
    player_dif = random.choice(['-1.5','-1','0','1','1.5'])

    detail = {
        "match_id": match_id,
        "teamA_players": json.dumps(teamA_selected, ensure_ascii=False),
        "teamB_players": json.dumps(teamB_selected, ensure_ascii=False),
        "teamA_headers": teamA_headers,
        "teamA_footers": teamA_footers,
        "teamA_hxg": teamA_hxg,
        "teamA_fxg": teamA_fxg,
        "teamB_headers": teamB_headers,
        "teamB_footers": teamB_footers,
        "teamB_hxg": teamB_hxg,
        "teamB_fxg": teamB_fxg,
        "minutes_played": 90,
        "match_state": match_state,
        "match_segment": match_segment,
        "player_dif": player_dif
    }
    return detail

def simula_match_breakdown(match_id, team_id, selected_players, is_home=True):
    """Genera datos fake para cada jugador de un equipo en un partido.
    Para cada jugador se simulan estadísticas como headers, footers, key passes, xG, etc.
    Se incluye además la influencia de tiros, goles esperados, acciones de sustitución y minutos jugados.
    """
    breakdown_rows = []
    # Ajuste de local/visitante: si en casa, se incrementa ligeramente algunos valores
    factor = 1.10 if is_home else 0.95

    for player in selected_players:
        player_data = get_player(team_id, player["id"])
        # Se basan en parámetros del jugador (base_spm y shooting) y se modifican aleatoriamente
        player_headers = int(random.randint(0, 3) * factor)
        player_footers = int(random.randint(1, 5) * factor)
        player_key_passes = random.randint(0, 3)
        player_non_assisted_footers = int(random.randint(0, 2) * factor)
        # xG y combinaciones (los valores de xG se basan en la habilidad; se usan rangos y se modifican aleatoriamente)
        player_hxg = round(random.uniform(0, player_data["shooting"]) * factor, 2)
        player_fxg = round(random.uniform(0, player_data["shooting"]) * factor, 2)
        player_kp_hxg = round(random.uniform(0,0.5), 2)
        player_kp_fxg = round(random.uniform(0,0.5), 2)
        player_hpsxg = round(random.uniform(0,0.3), 2)
        player_fpsxg = round(random.uniform(0,0.3), 2)
        gk_psxg = round(random.uniform(0,0.3), 2) if player_data["pos"]=="GK" else 0.0
        gk_ga = random.uniform(0,1) if player_data["pos"]=="GK" else 0
        player_sub_in = 0
        player_sub_out = 90
        in_status = random.choice(["starter", "substituted"])
        out_status = random.choice(["finished", "subbed"])
        player_fouls_committed = random.randint(0,3)
        player_fouls_drawn = random.randint(0,3)
        player_minutes_played = random.randint(60, 90)
        
        row = {
            "match_id": match_id,
            "player_id": player["id"],
            "player_headers": player_headers,
            "player_footers": player_footers,
            "player_key_passes": player_key_passes,
            "player_non_assisted_footers": player_non_assisted_footers,
            "player_hxg": player_hxg,
            "player_fxg": player_fxg,
            "player_kp_hxg": player_kp_hxg,
            "player_kp_fxg": player_kp_fxg,
            "player_hpsxg": player_hpsxg,
            "player_fpsxg": player_fpsxg,
            "gk_psxg": gk_psxg,
            "gk_ga": gk_ga,
            "player_sub_in": player_sub_in,
            "player_sub_out": player_sub_out,
            "in_status": in_status,
            "out_status": out_status,
            "player_fouls_committed": player_fouls_committed,
            "player_fouls_drawn": player_fouls_drawn,
            "player_minutes_played": player_minutes_played
        }
        breakdown_rows.append(row)
    return breakdown_rows
"""
Ahora generamos los datos y mostramos los INSERTs SQL
Se usarán tres listas para almacenar las instrucciones de Insert
"""
inserts_match_info = []
inserts_match_detail = []
inserts_match_breakdown = []

current_match_id = 1

# Para cada partido, se eligen dos equipos diferentes de la lista de equipos disponibles
team_ids = list(teams.keys())
for i in range(N_PARTIDOS):
    home_team = random.choice(team_ids)
    away_team = random.choice([tid for tid in team_ids if tid != home_team])

    def select_players(team_players):
        first_player = team_players[0]
        others = team_players[1:]
        random_4 = random.sample(others, 4)
        return [first_player] + random_4
    # Se obtienen los jugadores de cada equipo (solo sus nombres e id)
    teamA_players = [{"id": p["id"], "name": p["name"]} for p in teams[home_team]["players"]]
    teamB_players = [{"id": p["id"], "name": p["name"]} for p in teams[away_team]["players"]]
    teamA_selected = select_players(teamA_players)
    teamB_selected = select_players(teamB_players)

    # Generar match_info
    mi = simula_match_info(current_match_id, home_team, away_team)
    # Preparar la instrucción SQL para match_info (valores string se encierran entre comillas)
    sql_match_info = f"INSERT INTO finaltest.match_info (match_id, match_home_team_id, match_away_team_id, match_date, match_league_id, match_referee_id, match_total_fouls, match_yellow_cards, match_red_cards, minutes_played, home_elevation_dif, away_elevation_dif, home_travel, away_travel, home_rest_days, away_rest_days, home_importance, away_importance, temperature_c, is_raining, match_time) VALUES ({mi['match_id']}, {mi['match_home_team_id']}, {mi['match_away_team_id']}, '{mi['match_date']}', {mi['match_league_id']}, {mi['match_referee_id']}, {mi['match_total_fouls']}, {mi['match_yellow_cards']}, {mi['match_red_cards']}, {mi['minutes_played']}, {mi['home_elevation_dif']}, {mi['away_elevation_dif']}, {mi['home_travel']}, {mi['away_travel']}, {mi['home_rest_days']}, {mi['away_rest_days']}, {mi['home_importance']}, {mi['away_importance']}, {mi['temperature_c']}, {mi['is_raining']}, '{mi['match_time']}');"
    inserts_match_info.append(sql_match_info)

    # Generar match_detail
    md = simula_match_detail(current_match_id, home_team, away_team)
    sql_match_detail = ("INSERT INTO finaltest.match_detail (match_id, teamA_players, teamB_players, teamA_headers, teamA_footers, teamA_hxg, teamA_fxg, "
                        f"teamB_headers, teamB_footers, teamB_hxg, teamB_fxg, minutes_played, match_state, match_segment, player_dif) VALUES "
                        f"({md['match_id']}, '{md['teamA_players']}', '{md['teamB_players']}', {md['teamA_headers']}, {md['teamA_footers']}, {md['teamA_hxg']}, {md['teamA_fxg']}, "
                        f"{md['teamB_headers']}, {md['teamB_footers']}, {md['teamB_hxg']}, {md['teamB_fxg']}, {md['minutes_played']}, '{md['match_state']}', '{md['match_segment']}', '{md['player_dif']}');")
    inserts_match_detail.append(sql_match_detail)

    # Generar match_breakdown para ambos equipos
    breakdown_home = simula_match_breakdown(current_match_id, home_team, teamA_selected, is_home=True, )
    breakdown_away = simula_match_breakdown(current_match_id, away_team, teamB_selected, is_home=False)
    for row in breakdown_home + breakdown_away:
        sql_breakdown = ("INSERT INTO finaltest.match_breakdown (match_id, player_id, headers, footers, key_passes, non_assisted_footers, "
                        f"player_hxg, player_fxg, player_kp_hxg, player_kp_fxg, player_hpsxg, player_fpsxg, gk_psxg, gk_ga, player_sub_in, player_sub_out, "
                        f"in_status, out_status, player_fouls_committed, player_fouls_drawn, player_minutes_played) VALUES "
                        f"({row['match_id']}, '{row['player_id']}', {row['player_headers']}, {row['player_footers']}, {row['player_key_passes']}, {row['player_non_assisted_footers']}, "
                        f"{row['player_hxg']}, {row['player_fxg']}, {row['player_kp_hxg']}, {row['player_kp_fxg']}, {row['player_hpsxg']}, {row['player_fpsxg']}, "
                        f"{row['gk_psxg']}, {row['gk_ga']}, {row['player_sub_in']}, {row['player_sub_out']}, '{row['in_status']}', '{row['out_status']}', "
                        f"{row['player_fouls_committed']}, {row['player_fouls_drawn']}, {row['player_minutes_played']});")
        inserts_match_breakdown.append(sql_breakdown)

    current_match_id += 1
#Mostrar los INSERTs generados
for sql in inserts_match_info:
    print(sql)

for sql in inserts_match_detail:
    print(sql)

for sql in inserts_match_breakdown:
    print(sql)
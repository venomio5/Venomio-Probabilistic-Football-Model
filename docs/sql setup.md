## SQL Setup

CREATE DATABASE vpfm;
USE vpfm;

## Tables

### 1. match_info_summary
```
CREATE TABLE match_info_summary (
    match_id INT PRIMARY KEY,
    match_home_team_id INT,
    match_away_team_id INT,
    match_date DATETIME,
    match_league_id INT,
    match_referee_id INT,
    match_total_fouls INT DEFAULT 0,
    match_yellow_cards INT DEFAULT 0,
    match_red_cards INT DEFAULT 0
);
```
### 2. match_lineup_summary
```
CREATE TABLE match_lineup_summary (
    ras_id INT PRIMARY KEY,
    match_id INT,
    teamA_players JSON,
    teamB_players JSON,
    teamA_shots INT,
    teamB_shots INT,
    minutes_played INT
);
```
### 3. match_player_summary
```
CREATE TABLE match_player_summary (
    match_id INT,
    player_id INT,
    player_head_shots INT DEFAULT 0,
    player_foot_shots INT DEFAULT 0,
    player_key_passes INT DEFAULT 0,
    player_foot_non_assisted_shots INT DEFAULT 0,
    opponent_head_shots INT DEFAULT 0,
    opponent_foot_shots INT DEFAULT 0,
    opponent_head_xg FLOAT DEFAULT 0.0,
    opponent_foot_xg FLOAT DEFAULT 0.0,
    player_head_xg FLOAT DEFAULT 0.0,
    player_foot_xg FLOAT DEFAULT 0.0,
    player_kp_head_xg FLOAT DEFAULT 0.0,
    player_kp_foot_xg FLOAT DEFAULT 0.0,
    opponent_kp_head_xg FLOAT DEFAULT 0.0,
    opponent_kp_foot_xg FLOAT DEFAULT 0.0,
    player_sub_in INT DEFAULT 0,
    player_sub_out INT DEFAULT 0,
    in_status VARCHAR(50),
    out_status VARCHAR(50),
    player_fouls_committed INT DEFAULT 0,
    player_fouls_drawn INT DEFAULT 0,
    player_minutes_played INT DEFAULT 0,
    PRIMARY KEY (match_id, player_id)
);
```
### 4. players_data
```
CREATE TABLE players_data (
    player_id VARCHAR(20) PRIMARY KEY,
    player_name VARCHAR(100),
    current_team VARCHAR(50),
    player_pos VARCHAR(50),
    off_sh_coeff DECIMAL(6,3) DEFAULT NULL,
    def_sh_coeff DECIMAL(6,3) DEFAULT NULL
);
```
### 5. team_data
```
CREATE TABLE team_data (
    team_id INT PRIMARY KEY,
    team_name VARCHAR(100),
    team_elevation INT,
    team_coordinates VARCHAR(50),
    league VARCHAR(50)
);
```
## Notes


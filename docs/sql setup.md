## SQL Setup

CREATE DATABASE vpfm;
USE vpfm;

## Tables

### 1. match_info
```
CREATE TABLE match_info (
    match_id INT AUTO_INCREMENT PRIMARY KEY,
    match_home_team_id INT NOT NULL,
    match_away_team_id INT NOT NULL,
    match_date DATETIME NOT NULL,
    match_league_id INT,
    match_referee_id INT,
    match_total_fouls INT DEFAULT 0,
    match_yellow_cards INT DEFAULT 0,
    match_red_cards INT DEFAULT 0,
    minutes_played INT,
    home_elevation_dif FLOAT,
    away_elevation_dif FLOAT,
    home_travel FLOAT,
    away_travel FLOAT,
    home_rest_days INT,
    away_rest_days INT,
    home_importance BOOLEAN,
    away_importance BOOLEAN,
    temperature_c FLOAT,
    is_raining BOOLEAN,
    match_time VARCHAR(20),
    UNIQUE (match_home_team_id, match_away_team_id, match_date)
);
```
### 2. match_detail
```
CREATE TABLE match_detail (
    match_id INT NOT NULL,
    teamA_players JSON NOT NULL,
    teamB_players JSON NOT NULL,
    teamA_headers INT NOT NULL,
    teamA_footers INT NOT NULL,
    teamB_headers INT NOT NULL,
    teamB_footers INT NOT NULL,
    minutes_played INT NOT NULL,
    PRIMARY KEY (
        match_id,
        teamA_players,
        teamB_players,
        teamA_headers,
        teamA_footers,
        teamB_headers,
        teamB_footers,
        minutes_played
    ),
    CONSTRAINT fk_match_detail_id
        FOREIGN KEY (match_id) REFERENCES match_info (match_id)
        ON DELETE CASCADE
);
```
### 3. match_breakdown
```
CREATE TABLE match_breakdown (
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


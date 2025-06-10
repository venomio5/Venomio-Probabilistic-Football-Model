## SQL Setup

CREATE DATABASE vpfm;
USE vpfm;

## Tables

### match_info
```
CREATE TABLE match_info (
    match_id INT AUTO_INCREMENT PRIMARY KEY,
    home_team_id INT NOT NULL,
    away_team_id INT NOT NULL,
    date DATETIME NOT NULL,
    league_id INT,
    referee_name VARCHAR(100),
    total_fouls INT DEFAULT 0,
    yellow_cards INT DEFAULT 0,
    red_cards INT DEFAULT 0,
    home_elevation_dif INT,
    away_elevation_dif INT,
    home_travel INT,
    away_travel INT,
    home_rest_days INT,
    away_rest_days INT,
    temperature_c INT,
    is_raining BOOLEAN,
    url VARCHAR(200),
    UNIQUE (home_team_id, away_team_id, date)
);
```
### match_detail
```
CREATE TABLE match_detail (
    match_id INT NOT NULL,
    teamA_players JSON NOT NULL,
    teamB_players JSON NOT NULL,
    teamA_headers INT NOT NULL,
    teamA_footers INT NOT NULL,
    teamA_hxg FLOAT NOT NULL,
    teamA_fxg FLOAT NOT NULL,
    teamB_headers INT NOT NULL,
    teamB_footers INT NOT NULL,
    teamB_hxg FLOAT NOT NULL,
    teamB_fxg FLOAT NOT NULL,
    minutes_played INT NOT NULL,
    teamA_pdras FLOAT,
    teamB_pdras FLOAT,
    match_state ENUM('-1.5', '-1', '0', '1', '1.5') NOT NULL,
    match_segment ENUM('1', '2', '3', '4', '5', '6') NOT NULL,
    player_dif ENUM('-1.5', '-1', '0', '1', '1.5') NOT NULL,
    PRIMARY KEY (
        match_id,
        teamA_headers,
        teamA_footers,
        teamA_hxg,
        teamA_fxg,
        teamB_headers,
        teamB_footers,
        teamB_hxg,
        teamB_fxg,
        match_state,
        match_segment,
        player_dif
    ),
    CONSTRAINT fk_match_detail_id
        FOREIGN KEY (match_id) REFERENCES match_info (match_id)
        ON DELETE CASCADE
);
```
### match_breakdown
```
CREATE TABLE match_breakdown (
    match_id INT,
    player_id VARCHAR(50),
    headers INT DEFAULT 0,
    footers INT DEFAULT 0,
    key_passes INT DEFAULT 0,
    non_assisted_footers INT DEFAULT 0,
    hxg FLOAT DEFAULT 0.0,
    fxg FLOAT DEFAULT 0.0,
    kp_hxg FLOAT DEFAULT 0.0,
    kp_fxg FLOAT DEFAULT 0.0,
    hpsxg FLOAT DEFAULT 0.0,
    fpsxg FLOAT DEFAULT 0.0,
    gk_psxg FLOAT DEFAULT 0.0,
    gk_ga INT DEFAULT 0,
    sub_in INT DEFAULT 0,
    sub_out INT DEFAULT 0,
    in_status VARCHAR(50),
    out_status VARCHAR(50),
    fouls_committed INT DEFAULT 0,
    fouls_drawn INT DEFAULT 0,
    yellow_cards INT DEFAULT 0,
    red_cards INT DEFAULT 0,
    minutes_played INT DEFAULT 0,
    PRIMARY KEY (match_id, player_id),
    CONSTRAINT fk_match_breakdown_id
        FOREIGN KEY (match_id) REFERENCES match_info (match_id)
        ON DELETE CASCADE
);
```
### players_data
```
CREATE TABLE players_data (
    player_id VARCHAR(50) PRIMARY KEY,
    player_name VARCHAR(100),
    current_team VARCHAR(50),
    off_sh_coef FLOAT DEFAULT NULL,
    def_sh_coef FLOAT DEFAULT NULL,
    off_headers_coef FLOAT DEFAULT NULL,
    def_headers_coef FLOAT DEFAULT NULL,
    off_footers_coef FLOAT DEFAULT NULL,
    def_footers_coef FLOAT DEFAULT NULL,
    off_hxg_coef FLOAT DEFAULT NULL,
    def_hxg_coef FLOAT DEFAULT NULL,
    off_fxg_coef FLOAT DEFAULT NULL,
    def_fxg_coef FLOAT DEFAULT NULL,
    minutes_played INT DEFAULT 0,
    headers INT DEFAULT NULL,
    footers INT DEFAULT NULL,
    key_passes INT DEFAULT NULL,
    non_assisted_footers INT DEFAULT NULL,
    hxg FLOAT DEFAULT 0.0,
    fxg FLOAT DEFAULT 0.0,
    kp_hxg FLOAT DEFAULT 0.0,
    kp_fxg FLOAT DEFAULT 0.0,
    hpsxg FLOAT DEFAULT 0.0,
    fpsxg FLOAT DEFAULT 0.0,
    gk_psxg FLOAT DEFAULT 0.0,
    gk_ga INT DEFAULT 0,
    fouls_committed INT DEFAULT 0,
    fouls_drawn INT DEFAULT 0,
    yellow_cards INT DEFAULT 0,
    red_cards INT DEFAULT 0,
    sub_in JSON,
    sub_out JSON,
    in_status JSON,
    out_status JSON
);
```
### referee_data
```
CREATE TABLE referee_data (
    referee_name VARCHAR(100) PRIMARY KEY,
    fouls INT DEFAULT 0,
    yellow_cards INT DEFAULT 0,
    red_cards INT DEFAULT 0,
    minutes_played INT DEFAULT 0 
);
```
### shots_data
```
CREATE TABLE shots_data (
    shot_id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT,
    xg FLOAT NOT NULL,
    psxg FLOAT NOT NULL,
    shot_type ENUM('head', 'foot') NOT NULL,
    shooter_id VARCHAR(50) NOT NULL,
    assister_id VARCHAR(50) NOT NULL,
    GK_id VARCHAR(50) NOT NULL,
    off_players JSON NOT NULL,
    def_players JSON NOT NULL,
    total_PLSQA FLOAT DEFAULT NULL,
    shooter_SQ FLOAT DEFAULT NULL,
    assister_SQ FLOAT DEFAULT NULL,
    match_state ENUM('-1.5', '-1', '0', '1', '1.5') NOT NULL,
    match_segment ENUM('1', '2', '3', '4', '5', '6') NOT NULL,
    player_dif ENUM('-1.5', '-1', '0', '1', '1.5') NOT NULL,
    RSQ FLOAT DEFAULT NULL,
    shooter_A FLOAT DEFAULT NULL,
    GK_A FLOAT DEFAULT NULL,
    CONSTRAINT fk_shots_data_id
        FOREIGN KEY (match_id) REFERENCES match_info (match_id)
        ON DELETE CASCADE
);
```
### team_data
```
CREATE TABLE team_data (
    team_id INT AUTO_INCREMENT PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL UNIQUE,
    team_elevation INT NOT NULL,
    team_coordinates VARCHAR(50) NOT NULL,
    league_id INT NOT NULL,
    CONSTRAINT fk_league_id
        FOREIGN KEY (league_id) REFERENCES league_data(league_id)
        ON DELETE CASCADE
);
```
### league_data
```
CREATE TABLE league_data (
    league_id INT AUTO_INCREMENT PRIMARY KEY,
    league_name VARCHAR(100) UNIQUE NOT NULL,
    fbref_fixtures_url VARCHAR(200),
    last_updated_date DATE,
    is_active BOOLEAN,
    league_sg_url VARCHAR(200)
);
```
### schedule_data
```
CREATE TABLE schedule_data (
    schedule_id INT AUTO_INCREMENT PRIMARY KEY,
    home_team_id INT NOT NULL,
    away_team_id INT NOT NULL,
    date DATETIME NOT NULL,
    league_id INT,
    home_elevation_dif INT,
    away_elevation_dif INT,
    home_travel INT,
    away_travel INT,
    home_rest_days INT,
    away_rest_days INT,
    temperature_c INT,
    is_raining BOOLEAN,
    url VARCHAR(200),
    UNIQUE (home_team_id, away_team_id)
);
```
## Notes


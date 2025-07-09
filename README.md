# Venomio Probabilistic Football Model
**Version 7.0.1**
**Not distributed for public use or deployment.**

## Overview  
A sports terminal tool for sports trading or betting. 
*Explain later with images how to use it*
### Soccer Checklist
- **New Season**: Update league teams.
- **Every Week**: Update league data.
- **Every Day/Hour**: Update schedule.

## Soccer Algorithm Theory
Simulates each minute of a soccer match based on **Shots Per Minute (SPM)**. At each time step, each squad has a projected shots per minute (SPM) value. SPM changes dynamically based on:
- **Game state**: winning or losing (by 1 or more), or level.
- **Lineup changes**: substitutions or red cards.
- **Time segment**: 0-15, 15-30, 30-45, 45-60, 60-75, 75-90.

### Modeling Projected Shots Per Minute
* Data for the last preceding year.
#### Regularized Adjusted Shots (RAS) 
RAS is derived via **Ridge Regression** using:
- List of team A players
- List of team B players
- Total shots by team A
- Total shots by team B
- Total minutes played

The **Output** is each player's contribution to team shot production.

### Contextual XGBoost Model
RAS values serve as a baseline to an advanced XGBoost model, which integrates critical external and situational factors to predict final performance metrics (SPM). This are the features used:
- Total team RAS (As baseline shots per minute)
- Team_is_home (Bool)
- Team_elevation_dif & Opp_elevation_dif (stadium elevation - avg(league, team))
- Team_travel & Opp_travel (Distance in km)
- Team_rest_days & Opp_rest_days (Number of rest days)
- Match_state (-1.5, -1, 0, 1, 1.5)
- Match_segment (1, 2, 3, 4, 5, 6)
- Player_dif (-1.5, -1, 0, 1, 1.5)
- Temperature (°C) at kickoff
- Is_Raining (Bool)
- Match_time (aft, evening, night)

**Output**: Refined prediction of team-level and minute-level Shots Per Minute. Use high level minutes of RAS for certainty (Low minutes, the model is not as effective). 

# Shot Resolution
For each simulated shot:
## Regularized Adjusted Shot Type (RAST)
From the same **RAS** but for each type (Head or Foot):
- List of team A players
- List of team B players
- Total head and foot shots by team A
- Total head and foot shots by team B
- Total minutes played

## Specific players
- **Shooter**: Determined by weighted randomness favoring players with higher shot volume for the type.
- **Assister**: Determined by weighted randomness where headers receive full weight (100%), and foot depend on the shooting player’s ability to generate their own attempts, augmented by key passes (KP).

## Shot Quality
### Player-Level Shot Quality Attribution
Another regularized coefficient reflecting their average influence on xG when present.
- List of team A players
- List of team B players
- Total head and foot xg by team A
- Total head and foot xg by team B
- Per shot

**Output**: Each player's contribution to team shot quality.

### Refined Shot Quality Model
Aggregate the ridge data per shot and build a model to learn nonlinear, hierarchical patterns.
- Total PLSQA
- Shooter SQ
- Assister SQ
- Match_state
- Player_dif
- Target: xG

**Output**: Refined shot quality. 

### Player Performance Modifier
- RSQ
- Shooter Ability (Difference between xG and PSxG in %)
- GK Ability (Difference between PSxG and Goals in %)
- Team_is_home
- Team_elevation_dif
- Team_travel
- Team_rest_days
- Temperature_C
- Is_Raining
- Match_time (aft, evening, night)
- Target: Outcome of the Shot

**Output**: Refined post shot expected goal.

# Lineup Dynamics
## Substitutions
1. Pull historical substitution data for both teams from the database.
2. Compute how many subs each team usually makes in past games.
3. Based on how many substitutions each team can still make, determine how many they are realistically allowed to do now.
4. From historical data, find the most common minutes when each team usually makes subs.
5. Distribute the allowed number of substitutions across these likely minutes.
6. At each minute, check if it's a substitution minute.
7. If Yes – Do Substitution:
  - For players currently playing (active), calculate how likely each is to be subbed out. Factors: their total minutes played and match state.
  - Randomly pick players to be subbed out based on those weights.
  - For players on the bench (passive), calculate how likely each is to come in. Factors: their total minutes played and match state.
  - Randomly pick players to be subbed in based on those weights.
8. Remove chosen players from active, insert new ones from passive.

Repeat this process each time a substitution minute is reached.

## Card and Foul Simulation
1. Get the teams fouls per 90 minutes by getting the average of the team fouls comitted and the opponent fouls drawn.
2. Get the normalized teams fouls per 90 minutes by getting the average from the referee fouls per match, and the sum of the team and opponent fouls per 90 minutes. Then divide the teams fouls per 90 minute by the total average.
3. Multiply each normalized team fouls per minute with home and away factors, and team status (leading, trailing, level)
4. Choose on weighed  probability on who fouled, and then on weighed probability, choose between YC, RD, and None, based on referee data and player´s data. 

## Output Metrics
Minute-by-minute event log:
- Score.
- Players goals, assists, and red cards.

### SQL Setup
#### Database
CREATE DATABASE vpfm;
USE vpfm;

#### Tables
##### match_info
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
    away_travel INT,
    home_rest_days INT,
    away_rest_days INT,
    temperature_c INT,
    is_raining BOOLEAN,
    url VARCHAR(200),
    UNIQUE (home_team_id, away_team_id, date)
);
```
##### match_detail
```
CREATE TABLE match_detail (
    detail_id INT AUTO_INCREMENT PRIMARY KEY,
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
    UNIQUE KEY uq_match_detail (
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
    FOREIGN KEY (match_id) REFERENCES match_info (match_id)
        ON DELETE CASCADE
);
```
##### match_breakdown
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
    sub_in INT,
    sub_out INT,
    in_status VARCHAR(50),
    out_status VARCHAR(50),
    fouls_committed INT DEFAULT 0,
    fouls_drawn INT DEFAULT 0,
    yellow_cards INT DEFAULT 0,
    red_cards INT DEFAULT 0,
    minutes_played INT DEFAULT 0,
    PRIMARY KEY (match_id, player_id),
    FOREIGN KEY (match_id) REFERENCES match_info (match_id)
        ON DELETE CASCADE
);
```
##### players_data
```
CREATE TABLE players_data (
    player_id VARCHAR(50) PRIMARY KEY,
    current_team INT NOT NULL,
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
##### referee_data
```
CREATE TABLE referee_data (
    referee_name VARCHAR(100) PRIMARY KEY,
    fouls INT DEFAULT 0,
    yellow_cards INT DEFAULT 0,
    red_cards INT DEFAULT 0,
    matches_played INT DEFAULT 0 
);
```
##### shots_data
```
CREATE TABLE shots_data (
    shot_id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT,
    xg FLOAT NOT NULL,
    psxg FLOAT NOT NULL,
    outcome BOOLEAN,
    shot_type ENUM('head', 'foot') NOT NULL,
    shooter_id VARCHAR(50) NOT NULL,
    assister_id VARCHAR(50) NOT NULL,
    team_id INT NOT NULL,
    GK_id VARCHAR(50) NOT NULL,
    off_players JSON NOT NULL,
    def_players JSON NOT NULL,
    total_PLSQA FLOAT DEFAULT NULL,
    shooter_SQ FLOAT DEFAULT NULL,
    assister_SQ FLOAT DEFAULT NULL,
    match_state ENUM('-1.5', '-1', '0', '1', '1.5') NOT NULL,
    player_dif ENUM('-1.5', '-1', '0', '1', '1.5') NOT NULL,
    RSQ FLOAT DEFAULT NULL,
    shooter_A FLOAT DEFAULT NULL,
    GK_A FLOAT DEFAULT NULL,
    FOREIGN KEY (match_id) REFERENCES match_info (match_id)
        ON DELETE CASCADE
);
```
##### team_data
```
CREATE TABLE team_data (
    team_id INT AUTO_INCREMENT PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL,
    team_elevation INT NOT NULL,
    team_coordinates VARCHAR(50) NOT NULL,
    team_fixtures_url VARCHAR(200) NOT NULL,
    league_id INT NOT NULL,
    FOREIGN KEY (league_id) REFERENCES league_data(league_id)
        ON DELETE CASCADE,
    UNIQUE KEY uniq_team_league (team_name, league_id)
);
```
##### league_data
```
CREATE TABLE league_data (
    league_id INT AUTO_INCREMENT PRIMARY KEY,
    league_name VARCHAR(100) UNIQUE NOT NULL,
    fbref_fixtures_url VARCHAR(200),
    last_updated_date DATE,
    is_active BOOLEAN,
    ss_url VARCHAR(200),
    sh_baseline_coef FLOAT,
    hxg_baseline_coef FLOAT,
    fxg_baseline_coef FLOAT
);
```
##### schedule_data
```
CREATE TABLE schedule_data (
    schedule_id INT AUTO_INCREMENT PRIMARY KEY,
    home_team_id INT NOT NULL,
    away_team_id INT NOT NULL,
    date DATE NOT NULL,
    local_time TIME NOT NULL,
    venue_time TIME NOT NULL,
    league_id INT NOT NULL,
    home_elevation_dif INT,
    away_elevation_dif INT,
    away_travel INT,
    home_rest_days INT,
    away_rest_days INT,
    temperature INT,
    is_raining BOOLEAN,
    home_players_data JSON,
    away_players_data JSON,
    ss_url VARCHAR(200),
    referee_name VARCHAR(100),
    current_home_goals INT,
    current_away_goals INT,   
    current_period_start_timestamp INT,
    period VARCHAR(100),
    period_injury_time INT,
    last_minute_checked INT,
    simulate BOOLEAN,
    UNIQUE (home_team_id, away_team_id)
);
```
##### simulation_data
```
CREATE TABLE simulation_data (
    sim_id INT NOT NULL,
    schedule_id INT NOT NULL,
    minute INT NOT NULL,
    shooter VARCHAR(50) NOT NULL,
    squad VARCHAR(20) NOT NULL,
    outcome VARCHAR(20) NOT NULL,
    body_part VARCHAR(20) NOT NULL,
    assister VARCHAR(50),
    FOREIGN KEY (schedule_id) REFERENCES schedule_data(schedule_id)
        ON DELETE CASCADE
);
```

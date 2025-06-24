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
#### Regularized Adjusted Shots (RAS) 
RAS is derived via **Ridge Regression** using:
- List of team A players
- List of team B players
- Total shots by team A
- Total shots by team B
- Total minutes played

The **Output** is each player's contribution to team shot production.

* Data for the last preceding year.

### Contextual XGBoost Model
RAS values serve as core inputs to an advanced XGBoost model, which integrates critical external and situational factors to predict final performance metrics (SPM). 
| Feature             | Type          | Description                                                      |
|---------------------|---------------|------------------------------------------------------------------|
| Total RAS           | float         | Baseline shots per minute                                        |
| Team_id             | categorical   | Team identifier                                                  |
| Opp_id              | categorical   | Opponent identifier                                              |
| Team_is_home        | bool          | 1 = home, 0 = away                                               |
| Team_elevation_dif  | float         | Elevation difference (km): stadium elevation - avg(league, team) |
| Opp_elevation_dif   | float         | Elevation difference (km): stadium elevation - avg(league, team) |
| Team_travel         | float         | Travel distance (km)                                             |
| Opp_travel          | float         | Opponent travel distance (km)                                    |
| Team_rest_days      | int           | Team number of rest days                                         |
| Opp_rest_days       | int           | Opponent number of rest days                                     |
| Match_state         | categorical   | (-1.5, -1, 0, 1, 1.5)                                            |
| Match_segment       | categorical   | (1, 2, 3, 4, 5, 6)                                               |
| Player_dif          | categorical   | (-1.5, -1, 0, 1, 1.5)                                            |
| Team_importance     | bool (0/1)    | Final_Third_Critical (1 = yes, 0 = no)                           |
| Opp_importance      | bool (0/1)    | Final_Third_Critical (1 = yes, 0 = no)                           |
| Temperature_C       | float         | Temperature (°C) at kickoff                                      |
| Is_Raining          | bool          | 1 = yes, 0 = no                                                  |
| Match_time          | categorical   | (aft, evening, night)                                            |

**Output**: Refined prediction of team-level and minute-level Shots Per Minute. Use high level minutes of RAS for certainty (Low minutes, the model is not as effective). 

Data for the last 2 preceding years.

# Shot Resolution
For each simulated shot:
## Shot Type
Another **Ridge Regression** for each type (Head or Foot), normalize them and choose based on weighted randomness.  
| Feature           | Type        | Description      |
|-------------------|-------|------------------------|
| team_A_players    | JSON  | List of team A players |
| team_B_players    | JSON  | List of team B players |
| team_A_h/f_shots  | int   | Total h/f by team A    |
| team_B_h/f_shots  | int   | Total h/f by team B    |
| minutes_played    | int   | Total minutes played   |

## Specific players
- **Shooter**: Determined by weighted randomness favoring players with higher shot volume for the type.
- **Assister**: Determined by weighted randomness where headers receive full weight (100%), and foot depend on the shooting player’s ability to generate their own attempts, augmented by key passes (KP).

Add 1 to everyone for it to always have a probability.

## Shot Quality
### Player-Level Shot Quality Attribution
This is a base-level contextual impact. For each player: a regularized coefficient reflecting their average influence on xG when present.
| Feature           | Type        | Description      |
|-------------------|-------|------------------------|
| team_A_players    | JSON  | List of team A players |
| team_B_players    | JSON  | List of team B players |
| team_A_h/f_xg     | float | Total h/f xg by team A |
| team_B_h/f_xg     | float | Total h/f xg by team B |
| shots             | int   | Total shots            |

**Output**: Each player's contribution to team shot quality.

### Full-Factor Shot Quality Model
Aggregate the ridge data per shot and build a model to learn nonlinear, hierarchical patterns.
| Feature             | Type          | Description                                                      |
|---------------------|---------------|------------------------------------------------------------------|
| Total PLSQA         | float         | General shot quality for type of shot                            |
| Shooter SQ          | float         | Shooter shot quality for type of shot                            |
| Assister SQ         | float         | Assister shot quality for type of shot                           |
| Match_state         | categorical   | (-1.5, -1, 0, 1, 1.5)                                            |
| Match_segment       | categorical   | (1, 2, 3, 4, 5, 6)                                               |
| Player_dif          | categorical   | (-1.5, -1, 0, 1, 1.5)                                            |

**Output**: Refined shot quality. 

### Player Performance Modifier
| Feature             | Type          | Description                                                      |
|---------------------|---------------|------------------------------------------------------------------|
| FFSQ                | float         | Refined Shot Quality for type of shot                            |
| Shooter Ability     | float         | Difference between xG and PSxG in %                              |
| GK Ability          | float         | Difference between PSxG and Goals in %                           |
| Team_is_home        | bool          | 1 = home, 0 = away                                               |
| Team_elevation_dif  | float         | Elevation difference (km): stadium elevation - avg(league, team) |
| Team_travel         | float         | Travel distance (km)                                             |
| Team_rest_days      | int           | Team number of rest days                                         |
| Temperature_C       | float         | Temperature (°C) at kickoff                                      |
| Is_Raining          | bool          | 1 = yes, 0 = no                                                  |
| Match_time          | categorical   | (aft, evening, night)                                            |

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
### Fouls per Minute
| **Feature**                | **Type**    | **Description**                                        |
| -------------------------- | ----------- | ------------------------------------------------------ |
| `referee_id`               | Categorical | Encodes referee id                             |
| `team_id`                  | Categorical | Team committing the foul                               |
| `opp_id`                   | Categorical | Opposition team                                        |
| `is_home`                  | bool        | 1 if team is home                                      |
| `team_avg_fouls_committed` | Numeric     | Historical aggression level                            |
| `opp_avg_fouls_drawn`      | Numeric     | Tendency to provoke fouls                              |
| `referee_avg_fouls`        | Numeric     | General foul call rate by referee                      |
| `total_fouls`              | Numeric     | Total match fouls (optional context)                   |
| `minutes_played`           | Numeric     | Duration of the match (optional if embedded in target) |

**Target**: fouls_committed / minutes_played

### Card per Foul
2 for each card.
| **Feature**              | **Type**    | **Description**                       |
| ------------------------ | ----------- | ------------------------------------- |
| `referee_id`             | Categorical | Referee ID                            |
| `team_id`                | Categorical | Team receiving yellow                 |
| `opp_id`                 | Categorical | Opposition team                       |
| `is_home`                | Binary      | 1 if team is home                     |
| `team_cards_per_foul`    | Numeric     | Historical yellow-per-foul rate       |
| `referee_cards_per_foul` | Numeric     | Ref’s card strictness                 |
| `total_cards`            | Numeric     | Total cards in match (context)        |
| `total_fouls`            | Numeric     | Total fouls in match (optional)       |

**Target**: cards / fouls_committed
On red card (Or two yellows) → player removed, team plays short-handed.

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
    minutes_played INT DEFAULT 0 
);
```
##### shots_data
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
    FOREIGN KEY (match_id) REFERENCES match_info (match_id)
        ON DELETE CASCADE
);
```
##### team_data
```
CREATE TABLE team_data (
    team_id INT AUTO_INCREMENT PRIMARY KEY,
    team_name VARCHAR(100) NOT NULL UNIQUE,
    team_elevation INT NOT NULL,
    team_coordinates VARCHAR(50) NOT NULL,
    team_fixtures_url VARCHAR(200) NOT NULL,
    league_id INT NOT NULL,
    FOREIGN KEY (league_id) REFERENCES league_data(league_id)
        ON DELETE CASCADE
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
    league_sg_url VARCHAR(200)
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
    home_players JSON,
    away_players JSON,
    UNIQUE (home_team_id, away_team_id)
);
```
##### simulation_data
```
CREATE TABLE simulation_data (
    sim_id INT AUTO_INCREMENT PRIMARY KEY,
    match_id INT NOT NULL,
    minute INT NOT NULL,
    shooter VARCHAR(50) NOT NULL,
    squad VARCHAR(20) NOT NULL,
    outcome VARCHAR(20) NOT NULL,
    body_part VARCHAR(20) NOT NULL,
    assister VARCHAR(50),
    FOREIGN KEY (match_id) REFERENCES match_info(match_id)
        ON DELETE CASCADE
);
```

# Venomio Probabilistic Football Model  
**Version:** 7.0.1  

---

## Theoretical Framework

### Core Concept - Monte Carlo Simulation  
Simulates each minute of a soccer match through Shots Per Minute (SPM).  
- At every minute, each team has a projected SPM, dynamically adjusted by:  
  - Game state (winning, losing, level)  
  - Lineup changes (substitutions, red cards)  
  - Tactical modes (offensive, defensive, normal)

---

## Modeling Projected Shots Per Minute

### Regularized Adjusted Shots (RAS)  
- Derived via Ridge Regression  
- Input: Historical total team shots  
- Features: Players on field  
- Output: Player-level shot contribution  
- Time window: Past 1 year

### RAS XGBoost Model  
- RAS values feed into XGBoost for refined SPM prediction  
- Additional features:  
  - Baseline float: Total RAS per minute  
  - Team and opponent IDs  
  - Home status  
  - Elevation differences  
  - Travel distance  
  - Rest days  
  - Match state and segment  
  - Player differences  
  - Recent goals (last 10 minutes)  
  - Season and match time  
- Output: Minute-level refined team SPM  
- Time window: Past 2 years (higher minute counts yield better certainty)

---

## Shot Resolution

- Shot type determined (head or foot) based on player historical tendencies  
- Shooter and passer identified by shot volume and key passes  
- Shot quality calculated via historical xG data:  
  - Opponent defender xG averages  
  - Shooter and assister xG  
- Player finishing modifier: Difference between xG and PSxG  
- Goalkeeper performance modifier: Difference between PSxG and actual goals  
- Log each shot with responsible player/team and minute  

---

## Lineup Dynamics

### Substitutions  
- Modeled by manager timing, number of subs, and game state  

### Card and Foul Simulation  
- Each player has foul per minute rate  
- Each minute probabilistically determines foul occurrence  
- If foul occurs, referee-specific probabilities assign yellow/red cards  
- Red cards or double yellows remove player, team plays short-handed  

---

## Output Metrics

- Minute-by-minute event log including:  
  - Score and momentum (10-minute window)  
  - Goals, assists, red cards by players  

---

## Database Schema

```sql
CREATE DATABASE soccer_data;
USE soccer_data;

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

CREATE TABLE match_lineup_summary (
    ras_id INT PRIMARY KEY,
    match_id INT,
    teamA_players JSON,
    teamB_players JSON,
    teamA_shots INT,
    teamB_shots INT,
    minutes_played INT
);

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

CREATE TABLE players_data (
    player_id VARCHAR(20) PRIMARY KEY,
    player_name VARCHAR(100),
    current_team VARCHAR(50),
    player_pos VARCHAR(50),
    off_sh_coeff DECIMAL(6,3) DEFAULT NULL,
    def_sh_coeff DECIMAL(6,3) DEFAULT NULL
);

CREATE TABLE team_data (
    team_id INT PRIMARY KEY,
    team_name VARCHAR(100),
    team_elevation INT,
    team_coordinates VARCHAR(50),
    league VARCHAR(50)
);
```

## Feedback
Lineup percentage coverage metrics available

End of Documentation.

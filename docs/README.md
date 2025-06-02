# Venomio Probabilistic Football Model  
**Version 7.0.1**

## Theoretical Framework
### Core Concept - Monte Carlo
Simulates each minute of a soccer match based on **Shots Per Minute (SPM)**. At each time step, each squad has a projected shots per minute (SPM) value. SPM changes dynamically based on:
- **Game state**: winning or losing (by 1 or more), or level.
- **Lineup changes**: substitutions or red cards.
- **Time segment**: 0-15, 15-30, 30-45, 45-60, 60-75, 75-90.

### Modeling Projected Shots Per Minute
#### Regularized Adjusted Shots (RAS) 
RAS is derived via **Ridge Regression**:
| Feature           | Type        | Description      |
|-------------------|-------|------------------------|
| team_A_players    | JSON  | List of team A players |
| team_B_players    | JSON  | List of team B players |
| team_A_shots      | INT   | Total shots by team A  |
| team_B_shots      | INT   | Total shots by team B  |
| minutes_played    | INT   | Total minutes played   |

**Output**: Each player's contribution to team shot production.

Data for the last preceding year.

#### Contextual XGBoost Model
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

## Shot Resolution
For each simulated shot:
- **Shot Type**: Head or foot.
- **Specific players**:
  - **Shooter**: Determined by weighted randomness favoring players with higher shot volume for the type.
  - **Assister**: Determined by weighted randomness where headers receive full weight (100%), and foot depend on the shooting player’s ability to generate their own attempts, augmented by key passes (KP).
- **Shot Quality**: Based on historical XG data:
  - Defender opposition XG average (head/foot)
  - Shooter and assister XG quality (head/foot)
*tiros dividirlos por tactical_mode?
Player Finishing Performance Modifier: Based on the difference between xG  and PSxG.
Goalkeeper Performance Modifier: Based on the difference between PSxG and Goals. 

Log minute and player/team responsibles.

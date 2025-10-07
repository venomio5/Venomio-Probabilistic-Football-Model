# Venomio Probabilistic Football Model
**Version 8.0.1**

## Model Theory  
This model simulates each minute of a football (soccer) match with 2 outputs: Goal or No Goal. If there is a Goal, it changes the game state with the **Projected Expected Goals per Minute (PxG/M)** which is the **most important metric** in this model.

### Projected Expected Goals per Minute (PxG/M)
Projected Expected Goals per Minute (PxG/M) is calculated by:
#### Regularized Adjusted Expected Goals (RAxG)
A ridge regression (linear model) that learns individual players´ offensive and defensive impact. Having more weight on recent matches, for up to matches for the preceding year. Data is acquired by FBREF and input data to the model is:
- List of team A players
- List of team B players
- Total xG produced by team A
- Total xG produced by team B
- Total minutes played

Time decay for more weight for recent games.

#### Fatigue & Rythm
##### Initial_Fatigue
Initial Fatigue = min(1, Total_Decayed_Minutes / 90)

Total_Decayed_Minutes = Σ [Minutes_Played_in_Match * exp(-Days_Since_Match / 3)]

##### Rhythm Calculation
Rhythm = min(1, Total_Decayed_Minutes / 90)

Total_Decayed_Minutes = Σ [Minutes_Played_in_Match * exp(-Days_Since_Match / 14)]

##### In-Game Fatigue Accumulation
Fatigue(t) = Initial_Fatigue + (t / 45) × Max_Fatigue_Increase × Initial_Fatigue
Second half = Fatigue(t) = Fatigue_After_Halftime + ((t - 45) / 45) × Max_Fatigue_Increase × Fatigue_After_Halftime

Where:
Max_Fatigue_Increase = 1 (maximum 100% increase per half)
Fatigue accumulation scales with current fatigue level
Hard cap at 1.0 - cannot exceed complete exhaustion (Use a min(1,x)) to the toal fatigue

#####  In-Game Rhythm Improvement
Rhythm(t) = Rhythm_start + (1 - Rhythm_start) * Improvement_Rate * (1 - exp(-t / τ_inmatch))
grows quickly at first then asymptotes
Where:
Improvement_Rate = 0.2 (can improve up to 20% during a match)
τ_inmatch = 20

##### Halftime Recovery
Fatigue_After_Halftime = Fatigue_At_45min * 0.85 (15% recovery)

##### Final Performance Model
Effective_Off_RAxG = Base_Off_RAxG * (0.9 + 0.1 * Rhythm(t))
Effective_Def_RAxG = Base_Def_RAxG * (1 + ω * Fatigue(t))

Where:
- ω = 0.8 (fatigue hurts defense)

Rhythm affects offense, and fatigue affects defense.

#### Contextual XGBoost Model
*For this do a research beforehand for each if there is really an impact.
RAxG is then summed up to get the teams projected xG based on the RAxG alone. Then this value is added to an advanced XGBoost model, which integrates context awareness. This are the features used:
- Total team RAxG (As baseline xG per 90 minutes)
- Team_is_home = Bool
- Elevation_dif = (stadium elevation - avg(league, team)) 
- Travel_dif = Team_travel - Opp_travel (Distance in km). Home vs away = 0 - 500 = -500 | Away vs Home = 500 - 0 = 500.
- Match_state (-1.5, -0.5, 0, 0.5, 1.5) - if they are losing or wining by x goals.
- Player_dif (-1.5, -0.5, 0, 0.5, 1.5) - if a team has a red card advantage or disadvantage. 
- Temperature_dif = (°C at kickoff) (actual temp - avg(league, team)) 
- Is_Raining (Bool)
- Match_time (aft, evening, night)

The output is the final PxG/90. 

### Changes in PxG/M
Now, you may ask "Why going into a simulation of each minute of the game?", and the answer is: A soccer match is dynamic, losing or winning affect how players think, that affects performance. Not only that, but other things changes, and that chnages the following actions. So here are the things that changes the PxG/M:
#### Game state
Like I said, winning or losing (by 1 or more), or level. Each goal, changes the PxG/M of both teams. So the simualtions is tracking the score.
#### Lineup changes
In here there  are two things to keep in mind: subs and red cards.
##### Substitutions
The manager´s decision to sub in a player depends on the match state. Like I said, each game is different, not the same players will play evry time. I need a model to make the subs based on the managers history decisoins. Like this:
1. Pull historical substitution data for both teams from the database.
2. Compute how many subs each team usually makes in past h/a games.
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
##### Red Cards
Now obviously, there are a few players that are more probable to a red card, especially if already being on a yellow card. Again this is makes the game dynamic. So how will I simualte this? By:
1. Get the teams fouls per 90 minutes by getting the average of the team fouls comitted and the opponent fouls drawn.
2. Get the normalized teams fouls per 90 minutes by getting the average from the referee fouls per match, and the sum of the team and opponent fouls per 90 minutes. Then divide the teams fouls per 90 minute by the total average.
3. Multiply each normalized team fouls per minute with home and away factors, and team status (leading, trailing, level)
4. Choose on weighed  probability on who fouled, and then on weighed probability, choose between YC, RD, and None, based on referee data and player´s data. 
#### Time segment
As the game evolve, the fatigue increases, so the PxG/m differs. So at each time segment, there is a change in PxG/m: 0-15, 15-30, 30-45, 45-60, 60-75, 75-90.
#### Bayes' Theorem
Update the projections based on the real xG. For the live model.
#### Variance
Inlcude variance of a standard deviation for the projected goals per minute and other things.

### Model Checklist
- **New Season**: Update league teams.
- **Every Week**: Update league data.
- **Every Day/Hour**: Update schedule.
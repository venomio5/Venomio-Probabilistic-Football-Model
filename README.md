# Venomio Probabilistic Football Model

## Overview  
A Monte Carlo-based simulation model that predicts minute-by-minute Shots Per Minute (SPM) in soccer matches. Incorporates dynamic game states, lineup changes, and player performance to simulate realistic match events.

## Features  
- Minute-level simulation of shots based on team and player metrics  
- Dynamic adjustment of shot rates by game state (winning, losing, level) and lineup changes (subs, red cards)  
- Advanced shot quality estimation using xG and player finishing modifiers  
- Foul and card probabilistic simulation per player and referee profile  
- Detailed event logs: goals, assists, cards, momentum  
- SQL database schema for match, lineup, and player data storage  

## Model Components  
- **Regularized Adjusted Shots (RAS):** Player contribution to shots via ridge regression  
- **XGBoost:** Refines projected Shots Per Minute using RAS and contextual features  
- **Shot Resolution:** Assigns shot type, quality, shooter, and assister  
- **Lineup Dynamics:** Models substitutions and player dismissals  
- **Fouls & Cards:** Simulates fouls and referee decisions probabilistically  

## Purpose  
Intended as an internal analytical framework to support advanced football analytics and strategic modeling. Not distributed for public use or deployment.

## Requirements  
- Python with scikit-learn, XGBoost  
- SQL database (MySQL/PostgreSQL compatible)  

---

Strictly for controlled environments and expert application.

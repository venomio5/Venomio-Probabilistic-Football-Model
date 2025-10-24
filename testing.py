import core
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from tqdm import tqdm

def load_progress():
    """Load current progress from JSON file"""
    if os.path.exists('progress.json'):
        with open('progress.json', 'r') as f:
            return json.load(f)
    else:
        return {
            "current_date": "2021-04-16",
            "season_end": "2022-05-30"
        }

def save_progress(progress_data):
    """Save progress to JSON file"""
    with open('progress.json', 'w') as f:
        json.dump(progress_data, f, indent=2)

def save_teams_data():
    active_leagues_df = core.DB.select("SELECT * FROM leagues WHERE is_active = 1")

    pbar = tqdm(active_leagues_df["id"].tolist(), desc="Processing leagues", unit="league")
    for league_id in pbar:
        pbar.set_postfix({"league": core.get_league_name_by_id(league_id)})
        core.FillTeamsData(league_id)
    pbar.close()

def update_active_leagues(current_date):
    core.DB.execute("UPDATE leagues SET is_active = 0")

    activate_sql = """
    UPDATE leagues 
    SET is_active = 1 
    WHERE 
        %s >= DATE_SUB(last_updated_date, INTERVAL 7 DAY)
        AND %s <= last_updated_date
    """

    core.DB.execute(activate_sql, (current_date, current_date))

def main():
    """Simple automated data gathering"""
    initial_setup = False # CHANGE THIS ACCORDINGLY --------------------------------------------------- DELETE progress json if true and update the default dates
    progress = load_progress()
    current_date = datetime.strptime(progress["current_date"], "%Y-%m-%d").date()
    season_end = datetime.strptime(progress["season_end"], "%Y-%m-%d").date()

    if initial_setup:
        print("Starting initial setup...")
        save_teams_data()
        core.UpdateSchedule(from_date=current_date, to_date=season_end)
    else:
        print("Starting data gathering...")
        
        total_weeks = ((season_end - current_date).days // 7) + 1
        with tqdm(total=total_weeks, desc="Processing weeks", unit="week") as pbar:
            while current_date <= season_end:
                update_active_leagues(current_date)
                next_week = (pd.to_datetime(current_date) + timedelta(days=7))
                pbar.set_description(f"Processing {current_date}")
        
                print(f"\nScraping matches until {next_week}...")
                core.ScrapeMatchesData(to_date=next_week.strftime('%Y-%m-%d'), backtesting=True)
        
                print("Processing data...")
                core.ProcessData(current_date=pd.to_datetime(next_week))

                current_date = next_week.date()
                progress["current_date"] = current_date.strftime('%Y-%m-%d')

                save_progress(progress)
                
                pbar.update(1)

if __name__ == "__main__":
    main()
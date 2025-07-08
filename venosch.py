from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from datetime import datetime, timedelta
import core

scheduler = BackgroundScheduler()

def schedule_all_games():
    now = datetime.now()
    future = now + timedelta(hours=12)

    matches_df = core.DB.select("""
        SELECT *
        FROM schedule_data
        WHERE CONCAT(date, ' ', local_time) BETWEEN %s AND %s
    """, (now.strftime("%Y-%m-%d %H:%M:%S"), future.strftime("%Y-%m-%d %H:%M:%S")))

    for idx, row in matches_df.iterrows():
        local_time_str = (datetime.min + row["local_time"]).time().strftime("%H:%M:%S")
        game_time = datetime.strptime(f'{row["date"]} {local_time_str}', "%Y-%m-%d %H:%M:%S")
        trigger_time = game_time - timedelta(hours=2)

        if trigger_time < datetime.now():
            continue  # Ya pasó, no tiene sentido

        job_id = f'autolineup_{row["schedule_id"]}'

        if not scheduler.get_job(job_id):
            scheduler.add_job(
                func=execute_autolineup_once,
                trigger=DateTrigger(run_date=trigger_time),
                args=[row["schedule_id"], row["league_id"], f"{core.get_team_name_by_id(row['home_team_id'])} vs {core.get_team_name_by_id(row['away_team_id'])}"],
                id=job_id
            )

def execute_autolineup_once(schedule_id, league_id, title):
    core.AutoLineups(league_id, title)
    if check_players_exist(schedule_id):
        core.Alg(schedule_id)
    else:
        scheduler.add_job(
            func=retry_autolineup_until_players,
            trigger='interval',
            seconds=600,
            args=[schedule_id, league_id, title],
            id=f"retry_{schedule_id}",
            replace_existing=True
        )

def retry_autolineup_until_players(schedule_id, league_id, title):
    core.AutoLineups(league_id, title) 
    if check_players_exist(schedule_id):
        scheduler.remove_job(f"retry_{schedule_id}")
        core.Alg(schedule_id)

def check_players_exist(schedule_id):
    row = core.DB.select("""
        SELECT home_players, away_players
        FROM schedule_data
        WHERE schedule_id = %s
    """, (schedule_id,))

    if not row:
        return False

    home_players, away_players = row
    return bool(home_players) and bool(away_players)

schedule_all_games()

scheduler.start()

for job in scheduler.get_jobs():
    print(job)

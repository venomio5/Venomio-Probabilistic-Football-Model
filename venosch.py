from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from datetime import datetime, timedelta
import core
import multiprocessing as mp
import time

scheduler = None

def schedule_auto_lineups_info():
    now = datetime.now() - timedelta(hours=2)
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
            if not check_player_data_exist(int(row["schedule_id"])):
                execute_autolineup_once(int(row["schedule_id"]))
        else:
            job_id = f'autolineup_{int(row["schedule_id"])}'
            if not scheduler.get_job(job_id):
                scheduler.add_job(
                func=execute_autolineup_once,
                trigger=DateTrigger(run_date=trigger_time),
                args=[int(row["schedule_id"])],
                id=job_id
            )

        job_id = f'autolineup_{int(row["schedule_id"])}'

        if not scheduler.get_job(job_id):
            scheduler.add_job(
                func=execute_autolineup_once,
                trigger=DateTrigger(run_date=trigger_time),
                args=[int(row["schedule_id"])],
                id=job_id
            )

        start = max(datetime.now(), game_time)
        end = game_time + timedelta(hours=2.1)
        matchinfo_job_id = f"matchinfo_{int(row['schedule_id'])}"
        scheduler.add_job(
            func=process_match_info,
            trigger='interval',
            seconds=120,
            start_date=start,
            end_date=end,
            args=[int(row["schedule_id"])],
            id=matchinfo_job_id
        )

def execute_autolineup_once(schedule_id):
    core.AutoLineups(schedule_id)
    if check_player_data_exist(schedule_id):
        core.Alg(schedule_id, 0, 0, 0, 5, 5)
    else:
        scheduler.add_job(
            func=retry_autolineup_until_players,
            trigger='interval',
            seconds=600,
            args=[schedule_id],
            id=f"retry_{schedule_id}",
            replace_existing=True
        )

def retry_autolineup_until_players(schedule_id):
    core.AutoLineups(schedule_id) 
    if check_player_data_exist(schedule_id):
        scheduler.remove_job(f"retry_{schedule_id}")
        core.Alg(schedule_id, 0, 0, 0, 5, 5)

def process_match_info(schedule_id):
    core.AutoMatchInfo(schedule_id)

    df = core.DB.select("""
        SELECT  simulate,
                current_home_goals,
                current_away_goals,
                last_minute_checked,
                home_n_subs_avail,
                away_n_subs_avail
        FROM schedule_data
        WHERE schedule_id = %s
    """, (schedule_id,))

    if df.empty:
        return

    row = df.iloc[0]
    if row["simulate"] == 1:
        core.Alg(
            schedule_id,
            row["current_home_goals"],
            row["current_away_goals"],
            row["last_minute_checked"],
            row["home_n_subs_avail"],
            row["away_n_subs_avail"]
        )
        core.DB.execute("""
            UPDATE schedule_data
            SET simulate = 0
            WHERE schedule_id = %s
        """, (schedule_id,))

def check_player_data_exist(schedule_id):
    df = core.DB.select("""
        SELECT home_players_data, away_players_data
        FROM schedule_data
        WHERE schedule_id = %s
    """, (schedule_id,))

    if df.empty:
        return False

    row = df.iloc[0]
    home_players = row["home_players_data"]
    away_players = row["away_players_data"]
    return (home_players is not None and home_players != '') and (away_players is not None and away_players != '')

def main():
    global scheduler
    scheduler = BackgroundScheduler()

    schedule_auto_lineups_info()            
    scheduler.start()

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()

if __name__ == "__main__":
    mp.freeze_support()
    main()
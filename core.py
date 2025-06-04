import databasemanager

# ------------------------------ Fetch & Save data ------------------------------
# --------------- fds ---------------
# ------------------------------ Process data ------------------------------
# --------------- Fill Players Data  ---------------
class fill_players_data:
    def __init__(self):
        """
        Reset the table and fill the new players_data
        """
        db = databasemanager.DatabaseManager(host="localhost", user="root", password="venomio", database="finaltest")
        db.execute("TRUNCATE TABLE players_data;")
# ------------------------------ Monte Carlo ------------------------------
# ------------------------------ Trading ------------------------------
# Init

import core

# core.AutoMatchInfo(4475)
# core.AutoLineups(4522)

# core.AutoSS()

df= core.DB.select("SELECT * FROM users_data;")
print(df)
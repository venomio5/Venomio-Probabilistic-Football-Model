```
db = DatabaseManager(
    host="localhost",
    user="admin",
    password="secret",
    database="production"
)

# Select example
df = db.select("SELECT * FROM users WHERE status = %s", ("active",))

# Insert example
affected = db.execute("INSERT INTO logs (event) VALUES (%s)", ("startup",))

# Batch insert
affected = db.execute(
    "INSERT INTO metrics (key, value) VALUES (%s, %s)",
    [("cpu", 0.93), ("ram", 0.72)],
    many=True
)
```

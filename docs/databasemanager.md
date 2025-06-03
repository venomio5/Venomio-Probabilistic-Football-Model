Initializes a *MySQLConnectionPool* with UTF-8MB4 encoding and disabled autocommit for explicit transactional control.

# Public Methods
```
select(sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame
```
Executes a **SELECT** query and returns the result as a Pandas DataFrame.

Parameters:
- sql: Parameterized SQL query string.
- params: Optional sequence of parameters to bind.

Returns:
- pd.DataFrame: Result set with column labels derived from cursor metadata.

```
execute(sql: str, params: Sequence[Any] | None = None, many: bool = False) -> int
```
Executes **INSERT**, **UPDATE**, or **DELETE** operations with full transactional safety.

Parameters:
- sql: Parameterized SQL statement.
- params: Sequence of parameters or sequence of sequences (for executemany).
- many: If True, executes batch operations using executemany.

Returns:
- int: Count of rows affected.

# Usage Example
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

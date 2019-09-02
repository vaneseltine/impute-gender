import csv
import sqlite3

con = sqlite3.connect("./data/usnames.sqlite")
cur = con.cursor()
cur.execute("delete from usnames;")
# cur.execute(
#     "CREATE TABLE usnames (name, f int, m int, year int);"
# )

with open("./data/usnames.csv", "r") as fin:
    rdr = csv.reader(fin)
    _ = next(rdr)
    cur.executemany(
        """
    insert into usnames
        (name, f, m, year)
    values
        (?, ?, ?, ?);
    """,
        rdr,
    )
con.commit()
con.close()

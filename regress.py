import pickle
from collections import defaultdict
from csv import DictReader
from pathlib import Path
from pprint import pprint

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

print("bop")

"""
https://github.com/OpenGenderTracking/globalnamedata/raw/master/assets/usnames.csv
"""

CSV_IN = Path("./data/usnames.csv")
PICKLE_CACHE = Path("./usnames.pickle")

if not PICKLE_CACHE.exists():
    # if True:
    names_all = defaultdict(list)

    with CSV_IN.open() as infile:
        reader = DictReader(infile)
        for i, row in enumerate(reader):
            name = row.pop("Name")
            yr, n_f, n_m = (int(row[x]) for x in ["Year", "F", "M"])
            # print({"year": yr, "male": 0, "n": n_f})
            # exit()
            names_all[name].append({"year": yr, "male": 0, "n": n_f})
            names_all[name].append({"year": yr, "male": 1, "n": n_m})
    # pprint(names_all)

    with PICKLE_CACHE.open("wb") as handle:
        pickle.dump(names_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


with PICKLE_CACHE.open("rb") as handle:
    names_all = pickle.load(handle)

print(f"{len(names_all)} names")


for name in ["Matthew"]:  # , "Leslie", "Jordan"]:
    print(name)
    df = pd.DataFrame(names_all[name])
    glm = smf.glm(
        "male ~ year", family=sm.families.Binomial(), data=df, freq_weights=df["n"]
    )
    print(glm.fit().summary())
# >>> import csv
# >>> with open('names.csv', newline='') as csvfile:
# ...     reader = csv.DictReader(csvfile)
# ...     for row in reader:
# ...         print(row['first_name'], row['last_name'])

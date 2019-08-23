import pickle
from collections import defaultdict
from csv import DictReader
from pathlib import Path

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

print("bop")

"""
https://github.com/OpenGenderTracking/globalnamedata/raw/master/assets/usnames.csv
"""

CSV_IN = Path("./data/usnames.csv")
PICKLE_CACHE = Path("./data/usnames.pickle")

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
            names_all[name].append({"year": yr, "female": 1, "n": n_f})
            names_all[name].append({"year": yr, "female": 0, "n": n_m})
    # pprint(names_all)

    with PICKLE_CACHE.open("wb") as handle:
        pickle.dump(names_all, handle, protocol=pickle.HIGHEST_PROTOCOL)


with PICKLE_CACHE.open("rb") as handle:
    names_all = pickle.load(handle)

print(f"{len(names_all)} names")

print(next(iter(names_all.items())))
predict_years = pd.DataFrame([{"year": i} for i in range(1910, 2021, 10)])

for name in [
    "Matthew",
    "Matt",
    "Brooke",
    "Leslie",
    "Jordan",
    "Hillary",
    "Madison",
    "Monroe",
    "Skyler",
    "Charlie",
]:
    df = pd.DataFrame(names_all[name])
    grand_mean_year = (df["year"] * df["n"]).sum() / (df["n"].sum())
    df["year_cent"] = df["year"] - grand_mean_year
    predict_years["year_cent"] = predict_years["year"] - grand_mean_year
    n = df["n"].sum()
    n_female = df[df["female"] == 1]["n"].sum()
    n_male = n - n_female
    n_min = min(n_female, n_male)
    print(f"\n\nRunning {name} (n = {n}: {n_female} f, {n_male} m)")

    if n_min > 100000:
        formula = (
            "female ~ year_cent + pd.np.power(year_cent, 2) + pd.np.power(year_cent, 3)"
        )
    elif n_min > 5000:
        formula = "female ~ year_cent + pd.np.power(year_cent, 2)"
    elif n_min > 100:
        formula = "female ~ year_cent"
    else:  # may want to have a special case for n_min = 0 (e.g. Matthea)
        formula = "female ~ 1"  # intercept-only model
    glm = smf.glm(formula, family=sm.families.Binomial(), data=df, freq_weights=df["n"])
    print("\nFitting model...")
    results = glm.fit()
    print(results.summary())
    print("\n\nCreating prediction...")
    pred = results.get_prediction(predict_years)
    complete = predict_years.merge(
        pred.summary_frame(), left_index=True, right_index=True
    )
    print(complete)
# >>> import csv
# >>> with open('names.csv', newline='') as csvfile:
# ...     reader = csv.DictReader(csvfile)
# ...     for row in reader:
# ...         print(row['first_name'], row['last_name'])

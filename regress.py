import pickle
from collections import defaultdict
from csv import DictReader
from pathlib import Path
import urllib.request

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

pd.options.display.float_format = "{:.4f}".format

ORIGINAL_DATA = (
    "https://github.com/OpenGenderTracking/globalnamedata/raw/master/assets/usnames.csv"
)
CSV_IN = Path("./data/usnames.csv")


def load_data(csv_path):
    """
    Data are a lookup of {name: [sequence of female-n-year rows]} as:

        {
          'Aaron': [
              {'female': 1, 'n': 0, 'year': 1880},
              {'female': 0, 'n': 102, 'year': 1880},
              {'female': 1, 'n': 0, 'year': 1881},
              {'female': 0, 'n': 94, 'year': 1881},
              ...
              {'female': 1, 'n': 21, 'year': 2011},
              {'female': 0, 'n': 7593, 'year': 2011},
              {'female': 1, 'n': 21, 'year': 2012},
              {'female': 0, 'n': 7478, 'year': 2012}
              ],
          'Ab': [
              {'female': 1, 'n': 0, 'year': 1880},
              {'female': 0, 'n': 5, 'year': 1880},
              {'female': 1, 'n': 0, 'year': 1882},
              {'female': 0, 'n': 5, 'year': 1882},
              ...
              ]
          ...
        }

    """
    print(f"Loading data from {CSV_IN}...")
    pickle_path = ensure_pickle_cache(CSV_IN)

    with pickle_path.open("rb") as handle:
        all_data = pickle.load(handle)
    return all_data


def ensure_pickle_cache(csv_path):
    pickle_path = csv_path.with_suffix(".pickle")
    if pickle_path.exists():
        print(f"Already cached in {pickle_path}...")
    else:
        print(f"Caching {csv_path} into {pickle_path}...")
        cache_in_pickle(csv_path, pickle_path)
    if not pickle_path.exists():
        raise RuntimeError(f"Could not create {pickle_path}")
    return pickle_path


def cache_in_pickle(csv_path, pickle_path):
    dict_version = csv_to_dict(csv_path)
    with pickle_path.open("wb") as handle:
        pickle.dump(dict_version, handle, protocol=pickle.HIGHEST_PROTOCOL)


def csv_to_dict(csv_path):
    names_all = defaultdict(list)

    with csv_path.open() as infile:
        reader = DictReader(infile)
        for i, row in enumerate(reader):
            name = row.pop("Name")
            yr, n_f, n_m = (int(row[x]) for x in ["Year", "F", "M"])
            names_all[name].append({"year": yr, "female": 1, "n": n_f})
            names_all[name].append({"year": yr, "female": 0, "n": n_m})
    return names_all


def main():

    if not CSV_IN.exists():
        # Download the file from `url` and save it locally under `file_name`:
        print(f"Saving {ORIGINAL_DATA} as {CSV_IN}...")
        CSV_IN.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(ORIGINAL_DATA, CSV_IN)

    names_all = load_data(CSV_IN)

    from pprint import pprint

    pprint(f"{len(names_all)} names")

    j = iter(names_all.items())

    for name in [
        "Matthew",
        # "Matt",
        # "Brooke",
        "Leslie",
        # "Jordan",
        # "Hillary",
        # "Madison",
        # "Monroe",
        # "Charlie",
        # "Yiselle",
        # "Skyler",
    ]:
        print(f"\n\n{' '*110}Running {name}")
        if name not in names_all:
            print(f"Zero records of {name}")
            continue
        df = pd.DataFrame(names_all[name])
        # print(df)
        n = df["n"].sum()
        grand_mean_year = (df["year"] * df["n"]).sum() / n
        df["year_cent"] = df["year"] - grand_mean_year

        # min_year = df["year"].min()
        # max_year = df["year"].max()
        # year_range_for_prediction = range(min_year, max_year, 5)
        year_range_for_prediction = (1953, 1983, 2013)
        predict_years = pd.DataFrame([{"year": i} for i in year_range_for_prediction])

        predict_years["year_cent"] = predict_years["year"] - grand_mean_year
        n_female = df[df["female"] == 1]["n"].sum()
        n_male = n - n_female
        n_min = min(n_female, n_male)
        print(f"n = {n}: {n_female} f, {n_male} m")
        print(f"mean year {grand_mean_year}")

        formulae = [
            "female ~ C(year_cent)",
            "female ~ year_cent + pd.np.power(year_cent, 2) + pd.np.power(year_cent, 3)",
            "female ~ year_cent + pd.np.power(year_cent, 2)",
            "female ~ year_cent",
        ]

        # if n_min > 100000:
        #     formula = "female ~ year_cent + pd.np.power(year_cent, 2) + pd.np.power(year_cent, 3)"
        # elif n_min > 5000:
        #     formula = "female ~ year_cent + pd.np.power(year_cent, 2)"
        # elif n_min > 100:
        #     formula = "female ~ year_cent"
        # else:  # may want to have a special case for n_min = 0 (e.g. Matthea)
        #     formula = "female ~ 1"  # intercept-only model

        # if name == "Skyler":
        #     formula = "female ~ C(year_cent)"

        for formula in formulae:
            glm = smf.glm(
                formula, family=sm.families.Binomial(), data=df, freq_weights=df["n"]
            )
            print("Fitting model...")
            results = glm.fit()
            print(results.summary())
            print("Creating prediction...")
            try:
                pred = results.get_prediction(predict_years)
            except Exception as err:
                print(f"Could not run predictions ({err})")
                continue
            complete = predict_years.merge(
                pred.summary_frame(), left_index=True, right_index=True
            )
            print(complete)

            # back_in = complete.merge(df, on=["year"])
            # print(back_in)
    return df
    # >>> import csv
    # >>> with open('names.csv', newline='') as csvfile:
    # ...     reader = csv.DictReader(csvfile)
    # ...     for row in reader:
    # ...         print(row['first_name'], row['last_name'])


if __name__ == "__main__":
    x = main()

import pickle
import urllib.request
from collections import defaultdict
from csv import DictReader
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

OUTCOME = "female"
FEATURE_LIST = ["year"]

RANDOM_SEED = 48106
RUN_PLOTS = False
PREDICT_RANGE = False
SIMULTANEOUS_JOBS = 1
# Turning off matplotlib interactivity
plt.ioff()

# pd.options.display.float_format = "{:.4f}".format

ORIGINAL_DATA = (
    "https://github.com/OpenGenderTracking/globalnamedata/raw/master/assets/usnames.csv"
)
CSV_IN = Path("./data/usnames.csv")


def run_name(name, name_data, pred_from_x=None):
    if name not in name_data:
        print(f"\n{name}: UNKNOWN")
        return
    start_time = time()
    df = pd.DataFrame(name_data[name], columns=FEATURE_LIST + [OUTCOME])
    y_mean = df[OUTCOME].mean()
    y_mode = int(df[OUTCOME].mode())
    print(f"\n{name}: {y_mean*100:.1f}% female")

    labels = np.array(df[OUTCOME])
    features = np.array(df[FEATURE_LIST])

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=RANDOM_SEED
    )

    print(f"TOTAL n = {len(df):>12,}")
    print(f"TRAIN n = {len(x_train):>12,}")
    print(f" TEST n = {len(x_test):>12,}\n")

    # The baseline prediction is modal
    baseline_preds = np.array(y_mode)
    # Baseline errors, and display average baseline error
    baseline_err = np.mean(abs(baseline_preds - y_test))
    print(f"Mean baseline error: {baseline_err:.4f}")
    if baseline_err == 0.0:
        print("No variation in outcome. Skipping the rest...")
        return

    # Instantiate model with n_estimators # decision trees
    model = RandomForestClassifier(
        n_estimators=10, n_jobs=SIMULTANEOUS_JOBS, random_state=RANDOM_SEED
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Print out the mean absolute error (mean_abs_error)
    mean_abs_error = np.mean(errors)
    print(f"Mean absolute error: {mean_abs_error:.4f}")

    improved = abs(mean_abs_error - baseline_err)
    if improved:
        print(f"Improvement over baseline: {100*improved/baseline_err:.2f}%")

    print(f"     Train accuracy: {model.score(x_train, y_train):.4f}")
    print(f"      Test accuracy: {model.score(x_test, y_test):.4f}")

    if PREDICT_RANGE:
        for i in range(1920, 2011, 10):
            predict_from(model, df, i)

    if RUN_PLOTS:
        # Histogram predictions:
        fig, ax = plt.subplots()
        idx_female = np.where(y_test == 1)[0]
        idx_male = np.where(y_test == 0)[0]
        y_hat = model.predict_proba(x_test)
        ax.hist(
            y_hat[idx_female, 1],
            histtype="step",
            color="xkcd:sky blue",
            label="obs. female",
        )
        ax.hist(
            y_hat[idx_male, 1],
            histtype="step",
            color="xkcd:dusty orange",
            label="obs. male",
        )
        ax.set_xlabel("Predicted probability female")
        ax.set_ylabel("Observations")
        ax.set_title(name)
        plt.xlim([0, 1])
        plt.legend()
        fig.tight_layout()
        # plt.show()
        plt.savefig(f"output/{name}.png", pad_inches=0.2)
        plt.close()
    elapsed = time() - start_time
    print(f"{elapsed:.2f} sec.")
    if pred_from_x:
        return predict_from(model, df, pred_from_x)


def predict_from(model, df, i):
    probs = model.predict_proba([[i]])
    observed = df[df["year"] == i][OUTCOME]
    if observed.empty:
        actual = "-none-"
    else:
        actual = f"{(observed.sum() / observed.count())*100:.2f}%"
    print(i, f"true {actual} of {observed.count()}, pred: {probs}")
    return probs[0][-1]


def load_data(csv_path):
    print(f"Loading data from {CSV_IN}...")
    pickle_path = ensure_pickle_cache(CSV_IN)

    with pickle_path.open("rb") as handle:
        print(f"Loading data from {pickle_path}...")
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
            year = int(row["Year"])
            names_all[name].extend(
                int(row["F"]) * [(year, 1)] + int(row["M"]) * [(year, 0)]
            )
    return names_all


def main():

    if not CSV_IN.exists():
        # Download the file from `url` and save it locally under `file_name`:
        print(f"Saving {ORIGINAL_DATA} as {CSV_IN}...")
        CSV_IN.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(ORIGINAL_DATA, CSV_IN)

    name_data = load_data(CSV_IN)
    TEST_NAMES = [
        "Pat",
        "Brooke",
        "Matthew",
        "Leslie",
        "Jordan",
        "Monroe",
        "Skyler",
        "Matthea",
        "Raphael",
        "Zuwei",
        "Harold",
        "Dolores",
        "Natsuko",
        "Jinseok",
        "Zoran",
        "Zoe",
        "Ziyi",
        "Zi",
        "Zhou",
        "Mattheas",
    ]

    for name in TEST_NAMES:
        run_name(name, name_data)
    return name_data


if __name__ == "__main__":
    name_data = main()

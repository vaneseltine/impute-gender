import csv
import logging
import pickle
import sys
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
RUN_PLOTS = True
PREDICT_RANGE = False
SIMULTANEOUS_JOBS = 1
# Turning off matplotlib interactivity
plt.ioff()

# pd.options.display.float_format = "{:.4f}".format

OUTPUT_DIR = Path("./output/")
ORIGINAL_DATA = (
    "https://github.com/OpenGenderTracking/globalnamedata/raw/master/assets/usnames.csv"
)
CSV_IN = Path("./data/usnames.csv")


def get_package_logger(std_out=True, log_file=None, debug=True):

    log_level = logging.DEBUG if debug else logging.INFO

    if debug:
        formatter = logging.Formatter(
            fmt="{asctime} {name:<10} {lineno:>3}:{levelname:<7} | {message}",
            style="{",
            datefmt=r"%Y%m%d-%H%M%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="{asctime} {levelname:>7} | {message}", style="{", datefmt=r"%H:%M:%S"
        )

    handlers = []
    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    if std_out:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        handlers.append(stream_handler)
    logging.basicConfig(level=logging.INFO, handlers=handlers)

    logging.getLogger().setLevel(log_level)
    new_logger = logging.getLogger(__name__)
    new_logger.debug(f"Logging from {__name__}, {__file__}")
    return new_logger


logger = get_package_logger()

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def run_name(name, name_data, ad_hoc_years=None, run_all_years=False):
    if name not in name_data:
        logger.debug("")
        logger.debug(f"{name}: UNKNOWN")
        return
    start_time = time()
    df = pd.DataFrame(name_data[name], columns=FEATURE_LIST + [OUTCOME])
    y_mean = df[OUTCOME].mean()
    y_mode = int(df[OUTCOME].mode())
    logger.debug(f"{name}: {y_mean*100:.1f}% female")

    labels = np.array(df[OUTCOME])
    features = np.array(df[FEATURE_LIST])

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=RANDOM_SEED
    )

    logger.debug(f"TOTAL n = {len(df):>12,}")
    logger.debug(f"TRAIN n = {len(x_train):>12,}")
    logger.debug(f" TEST n = {len(x_test):>12,}")

    # The baseline prediction is modal
    baseline_preds = np.array(y_mode)
    # Baseline errors, and display average baseline error
    baseline_err = np.mean(abs(baseline_preds - y_test))
    logger.debug(f"Mean baseline error: {baseline_err:.4f}")
    if baseline_err == 0.0:
        logger.debug("No variation in outcome. Skipping the rest...")
        return [[name, None, y_mean]]

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
    logger.debug(f"Mean absolute error: {mean_abs_error:.4f}")

    improved = abs(mean_abs_error - baseline_err)
    if improved:
        logger.debug(f"Improvement over baseline: {100*improved/baseline_err:.2f}%")

    logger.debug(f"     Train accuracy: {model.score(x_train, y_train):.4f}")
    logger.debug(f"      Test accuracy: {model.score(x_test, y_test):.4f}")

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
    logger.debug(f"{elapsed:.2f} sec.")
    if ad_hoc_years:
        for yr in ad_hoc_years:
            predict_from(model, df, yr)
    if run_all_years:
        collection = []
        for yr in range(1902, 2013):
            pred_f = predict_from(model, df, yr)
            collection.append([name, yr, pred_f])
        return collection


def predict_from(model, df, i):
    probs = model.predict_proba([[i]])
    observed = df[df["year"] == i][OUTCOME]
    if observed.empty:
        actual = "n/a"
        # result = None
    else:
        actual = f"{(observed.sum() / observed.count())*100:.1f}%"
    result = probs[0][-1]
    logger.debug(
        f"{i}: n = {observed.count():<8} obs: {actual:<7} pred: {result*100:.1f}%"
    )
    return result


def load_data(csv_path):
    logger.debug(f"Loading data from {CSV_IN}...")
    pickle_path = ensure_pickle_cache(CSV_IN)

    with pickle_path.open("rb") as handle:
        logger.debug(f"Loading data from {pickle_path}...")
        all_data = pickle.load(handle)
    return all_data


def ensure_pickle_cache(csv_path):
    pickle_path = csv_path.with_suffix(".pickle")
    if pickle_path.exists():
        logger.debug(f"Already cached in {pickle_path}...")
    else:
        logger.debug(f"Caching {csv_path} into {pickle_path}...")
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


TEST_NAMES = [
    "Pat",
    "Brooke",
    "Tristan",
    # "Matthew",
    # "Leslie",
    # "Jordan",
    # "Monroe",
    # "Skyler",
    # "Matthea",
    # "Raphael",
    # "Zuwei",
    # "Harold",
    # "Dolores",
    # "Natsuko",
    # "Jinseok",
    # "Zoran",
    # "Zoe",
    # "Ziyi",
    # "Zi",
    # "Zhou",
    # "Mattheas",
]


def main():

    if len(sys.argv) > 1:
        test_names = sys.argv[1:]
    else:
        test_names = TEST_NAMES

    if not CSV_IN.exists():
        # Download the file from `url` and save it locally under `file_name`:
        logger.debug(f"Saving {ORIGINAL_DATA} as {CSV_IN}...")
        CSV_IN.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(ORIGINAL_DATA, CSV_IN)

    name_data = load_data(CSV_IN)

    for name in test_names:
        collection = run_name(name, name_data, ad_hoc_years=None, run_all_years=True)
        if not collection:
            logger.debug(f"No data for {name}; producing no output.")
            continue
        output_file = (OUTPUT_DIR / "matvan" / name.lower()).with_suffix(".csv")
        with output_file.open("w", newline="") as outfile:

            wrtr = csv.writer(outfile)
            wrtr.writerow(["name", "year", "pred_f"])

            for name, yr, pred_f in collection:

                formatted_pred = None if pred_f is None else round(pred_f, 5)
                row = [name.lower(), yr, formatted_pred]
                wrtr.writerow(row)
                print(f"CSV: {row}")


if __name__ == "__main__":
    name_data = main()

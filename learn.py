from collections import defaultdict
from csv import DictReader
from pathlib import Path
from time import time

import forestci as fci
import numpy as np
import pandas as pd
import pydot  # Pull out one tree from the forest
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from matplotlib import pyplot as plt

csv_path = Path("./data/usnames.csv")

names_all = defaultdict(list)

with csv_path.open() as infile:
    reader = DictReader(infile)
    for i, row in enumerate(reader):
        name = row.pop("Name")
        year = int(row["Year"])
        names_all[name].extend(
            int(row["F"]) * [(year, 1)] + int(row["M"]) * [(year, 0)]
        )

# with csv_path.open() as infile:
#     reader = DictReader(infile)
#     for i, row in enumerate(reader):
#         name = row.pop("Name")
#         yr, n_f, n_m = (int(row[x]) for x in ["Year", "F", "M"])
#         if n_f > 0:
#             names_all[name].append({"year": yr, "female": 1, "n": n_f})
#         if n_m > 0:
#             names_all[name].append({"year": yr, "female": 0, "n": n_m})

OUTCOME = "female"
FEATURE_LIST = ["year"]


def run_name(name):
    if name not in names_all:
        print(name, "UNKNOWN")
        return
    start_time = time()
    df = pd.DataFrame(names_all[name], columns=FEATURE_LIST + [OUTCOME])
    y_mean = df[OUTCOME].mean()
    y_mode = int(df[OUTCOME].mode())
    print(f"\n{name}: {y_mean*100:.1f}% female")  # (mode = {y_mode})")

    # print(df)
    # print(df.sample(frac=0.5, random_state=42))

    labels = np.array(df[OUTCOME])
    features = np.array(df[FEATURE_LIST])
    # print(df)
    # print(features)
    # sample_weights = np.array(df["n"])

    # print(sample_weights)

    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.25  # , random_state=42
    )

    # print("x_train", x_train[:5])
    # print("x_test", x_test[:5])
    # print("y_train", y_train[:5])
    # print("y_test", y_test[:5])
    # print("sw_train", sw_train[:5])
    # print("sw_test", sw_test[:5])

    # stacked = pd.DataFrame(np.hstack((x_test[0], y_test, sw_test)))
    # print(stacked)
    # print("x_train", x_train, len(x_train))
    # print("x_test", x_test, len(x_test))
    # print("y_train", y_train, len(y_train))
    # print("y_test", y_test, len(y_test))

    print(f"TOTAL n = {len(df):>12,}")
    print(f"TRAIN n = {len(x_train):>12,}")
    print(f" TEST n = {len(x_test):>12,}\n")
    # print("Training Features Shape:", x_train.shape)
    # print("Training Labels Shape:", y_train.shape)
    # print("Testing Features Shape:", x_test.shape)
    # print("Testing Labels Shape:", y_test.shape)

    # The baseline prediction is modal
    baseline_preds = np.array(y_mode)
    # Baseline errors, and display average baseline error
    baseline_errors = abs(baseline_preds - y_test)
    mbe = np.mean(baseline_errors)
    print(f"Mean baseline error: {mbe:.4f}")
    # Average baseline error:  0.3

    # Instantiate model with 1000 decision trees
    # rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model = RandomForestClassifier(n_estimators=10, n_jobs=6)
    # Train the model on training data
    model.fit(x_train, y_train)  # , sample_weight=sw_train)

    # Use the forest's predict method on the test data
    predictions = model.predict(x_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Print out the mean absolute error (mae)
    mae = np.mean(errors)
    print(f"Mean absolute error: {mae:.4f}")

    raw_improvement = abs(mae - mbe)
    if raw_improvement:
        print(f"Improvement over baseline: {100*raw_improvement/mbe:.2f}%")
    else:
        print("Identical predictions to baseline.")

    # , sample_weight=sw_train)
    print(f"     Train accuracy: {model.score(x_train, y_train):.4f}")
    # , sample_weight=sw_test)
    print(f"      Test accuracy: {model.score(x_test, y_test):.4f}")

    for i in range(1920, 2011, 10):
        probs = model.predict_proba([[i]])
        observed = df[df["year"] == i][OUTCOME]
        if observed.empty:
            actual = "-none-"
        else:
            actual = f"{(observed.sum() / observed.count())*100:.2f}%"
        print(i, f"observed {actual}, pred. {probs}")

    # # Histogram predictions without error bars:
    # fig, ax = plt.subplots(1)

    # idx_female = np.where(y_test == 1)[0]
    # idx_male = np.where(y_test == 0)[0]
    # y_hat = model.predict_proba(x_test)

    # ax.hist(y_hat[idx_female, 1], histtype="step", label="female")
    # ax.hist(y_hat[idx_male, 1], histtype="step", label="male")
    # ax.set_xlabel("Prediction (spam probability)")
    # ax.set_ylabel("Number of observations")
    # plt.legend()

    # # Calculate the variance
    # V_IJ_unbiased = fci.random_forest_error(model, x_train, x_test)

    # # Plot forest prediction for emails and standard deviation for estimates
    # # Blue points are spam emails; Green points are non-spam emails
    # fig, ax = plt.subplots(1)
    # ax.scatter(y_hat[idx_female, 1], np.sqrt(V_IJ_unbiased[idx_female]), label="female")

    # ax.scatter(y_hat[idx_male, 1], np.sqrt(V_IJ_unbiased[idx_male]), label="male")

    # ax.set_xlabel("Prediction (spam probability)")
    # ax.set_ylabel("Standard deviation")
    # plt.legend()
    # plt.show()
    # print(x_train)
    # print(y_train)

    # exit()

    # for metric in (
    #     metrics.accuracy_score,
    #     metrics.precision_score,
    #     metrics.recall_score,
    #     metrics.f1_score,
    # ):
    #     print(f"Train {metric} {metric(x_train, y_train):.4f}")
    #     # , sample_weight=sw_test)
    #     print(f" Test {metric} {metric(x_test, y_test):.4f}")

    # print("test", y_test[:15])
    # print("pred", predictions[:15])

    elapsed = time() - start_time
    print(f"{elapsed:.2f} sec.")


# def make_tree(model):
#     tree = model.estimators_[5]  # Export the image to a dot file
#     export_graphviz(
#         tree, out_file="tree.dot", feature_names=feature_list, rounded=True, precision=1
#     )  # Use dot file to create a graph
#     (graph,) = pydot.graph_from_dot_file("tree.dot")  # Write graph to a png file
#     graph.write_png("tree.png")

# def examine_important_features(model):
#     # Get numerical feature importances
#     importances = list(
#         model.feature_importances_
#     )  # List of tuples with variable and importance
#     feature_importances = [
#         (feature, round(importance, 2))
#         for feature, importance in zip(feature_list, importances)
#     ]  # Sort the feature importances by most important first
#     feature_importances = sorted(
#         feature_importances, key=lambda x: x[1], reverse=True
#     )  # Print out the feature and importances
#     [print("Variable: {:20} Importance: {}".format(*pair)) for pair in feature_importances]


for name in [
    # "Matthew",
    # "Leslie",
    # "Jordan",
    # "Monroe",
    # "Skyler",
    # "Matthea",
    # "Raphael",
    # "Zuwei",
    # "Harold",
    # "Pat",
    "Dolores",
    "Natsuko",
    "Jinseok",
    "Zoran",
    "Zoe",
    "Ziyi",
    "Zi",
    "Zhou",
    "Mattheas",
]:
    run_name(name)

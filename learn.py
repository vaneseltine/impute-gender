from collections import defaultdict
from csv import DictReader
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import pydot  # Pull out one tree from the forest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz

csv_path = Path("./data/leslies.csv")

names_all = defaultdict(list)

# with csv_path.open() as infile:
#     reader = DictReader(infile)
#     for i, row in enumerate(reader):
#         name = row.pop("Name")
#         year = int(row["Year"])
#         names_all[name].extend(
#             int(row["F"]) * [(year, 1)] + int(row["M"]) * [(year, 0)]
#         )

with csv_path.open() as infile:
    reader = DictReader(infile)
    for i, row in enumerate(reader):
        name = row.pop("Name")
        yr, n_f, n_m = (int(row[x]) for x in ["Year", "F", "M"])
        names_all[name].append({"year": yr, "female": 1, "n": n_f})
        names_all[name].append({"year": yr, "female": 0, "n": n_m})


for name in ["Leslie", "Matthew"]:
    start_time = time()
    # print(names_all)
    df = pd.DataFrame(names_all[name], columns=["year", "female", "n"])
    # print(df)
    print(name)
    print("n =", len(df))
    y_mean = df["female"].mean()
    print("mean =", y_mean)
    y_mode = 1  # int(df["female"].mode())
    print("mode =", y_mode)

    labels = np.array(df["female"])
    feature_list = ["year"]
    features = np.array(df[feature_list])
    print(df)
    sample_weights = df["n"]
    print(sample_weights)

    x_train, x_test, y_train, y_test, sw_train, sw_test = train_test_split(
        features, labels, sample_weights, test_size=0.25, random_state=42
    )
    # print("x_train", x_train, len(x_train))
    # print("x_test", x_test, len(x_test))
    # print("y_train", y_train, len(y_train))
    # print("y_test", y_test, len(y_test))

    print(f"Training n = {len(x_train)}; testing n = {len(x_test)}")
    # print("Training Features Shape:", x_train.shape)
    # print("Training Labels Shape:", y_train.shape)
    # print("Testing Features Shape:", x_test.shape)
    # print("Testing Labels Shape:", y_test.shape)

    # The baseline prediction is modal
    baseline_preds = np.array(y_mode)
    # Baseline errors, and display average baseline error
    baseline_errors = abs(baseline_preds - y_test)
    print("Mean baseline error: ", round(np.mean(baseline_errors), 2))
    # Average baseline error:  0.3

    # Instantiate model with 1000 decision trees
    # rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model = RandomForestClassifier(n_estimators=2, n_jobs=6, random_state=42)
    # Train the model on training data
    model.fit(x_train, y_train, sample_weight=sw_train)

    # Use the forest's predict method on the test data
    predictions = model.predict(x_test)
    # Calculate the absolute errors
    errors = abs(predictions - y_test)
    # Print out the mean absolute error (mae)
    print("Mean absolute error:", round(np.mean(errors), 2))

    print("Train accuracy", accuracy_score(y_train, model.predict(x_train)))
    print("Test accuracy", accuracy_score(y_test, predictions))

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

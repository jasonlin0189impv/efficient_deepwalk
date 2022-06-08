import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier


def evaluation(embed_model, label, test_size, metric="f1"):
    assert metric in ["f1", "acc"], "Only support 'f1' and 'acc'!"
    embed = embed_model.get_embeds()
    embed_pd = pd.DataFrame.from_dict(embed, orient="index")
    embed_pd.index = embed_pd.index.astype(int)

    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, max_depth=5))
    results_train = []
    results_test = []

    for seed in range(5):
        X_train, X_test, y_train, y_test = train_test_split(
            embed_pd, label, test_size=test_size, random_state=seed
        )
        clf.fit(X_train, y_train)

        if metric == "f1":
            results_train.append(
                f1_score(y_train, clf.predict(X_train), average="micro")
            )
            results_test.append(f1_score(y_test, clf.predict(X_test), average="micro"))
        else:
            results_train.append(accuracy_score(y_train, clf.predict(X_train)))
            results_test.append(accuracy_score(y_test, clf.predict(X_test)))

    print(
        f"Training {metric}: ",
        np.mean(results_train),
        f"Testing {metric}",
        np.mean(results_test),
    )

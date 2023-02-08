import gzip
import pickle
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sys
datasets_names = ["Matek-19", "INT-20","Acevedo-20"]

path = 'Features_Files/'


AE = "AE_CFE.dat"

with gzip.open(os.path.join(path, AE),"rb") as f:
    tdata = pickle.load(f)[0]

split = [None] * len(datasets_names)
data = [None] * len(datasets_names)
for i, dataset in enumerate(datasets_names):
    data[i] = [x for x in tdata if x["dataset"] == datasets_names.index(dataset)]
    X_train, X_test, _, _ = train_test_split(data[i], data[i], test_size=0.20, random_state=42)
    split[i] = X_train, X_test


scores = np.zeros((len(datasets_names),len(datasets_names)))

for i, dataset in enumerate(datasets_names):
    train_z_list = [x["z"].squeeze() for x in split[i][0]]
    train_label_list = [x["label"] for x in split[i][0]]

    clf = RandomForestClassifier(max_depth=16, random_state=0)
    clf.fit(train_z_list, train_label_list)

    test_z_list = [x["z"].squeeze() for x in split[i][1]]
    test_label_list = [x["label"] for x in split[i][1]]
    scores[i][i] = clf.score(test_z_list, test_label_list)
    for j, dataset in enumerate(datasets_names):
        if j == i:
            continue
        z_list = np.concatenate([[x["z"].squeeze() for x in split[j][0]],[x["z"].squeeze() for x in split[j][1]]])
        label_list = np.concatenate([[x["label"] for x in split[j][0]],[x["label"] for x in split[j][1]]])
        scores[i][j] = clf.score(z_list, label_list)

print(datasets_names)
print(scores)


print("done")

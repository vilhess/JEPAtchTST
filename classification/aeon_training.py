import numpy as np
from aeon.classification.hybrid import HIVECOTEV2
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

def training_module(name, trainset):
    
    if name=="KNN_DTW":
        model = KNeighborsTimeSeriesClassifier(distance="dtw")
    elif name=="HIVECOTEV2":
        model = HIVECOTEV2()

    values, labels = [], []
    for val, lab in trainset:
        values.extend(val.permute(0, 2, 1).tolist())
        labels.extend(lab.tolist())
    values = np.asarray(values)
    model.fit(values, labels)

    return model

def test_module(model, testset):
    values = []
    all_targets = []
    for val, label in testset:
        values.extend(val.permute(0, 2, 1).tolist())
        all_targets.extend(label.tolist())
    values = np.asarray(values)
    all_preds = model.predict(values)
    return all_preds, all_targets
    
"""Classical baseline (Section IV-C): a 16-8 ReLU MLP on the standardized raw features,
trained with Adam (learning rate 1e-3) for 200 epochs. It involves no measurement,
so it is shot-free."""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def mlp_accuracy(Xtr, ytr, Xte, yte, epochs=200, seed=0):
    scaler = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", solver="adam",
                        learning_rate_init=1e-3, max_iter=1, warm_start=True, random_state=seed)
    for _ in range(epochs):                 # one Adam epoch per call
        clf.fit(scaler.transform(Xtr), ytr)
    return float((clf.predict_proba(scaler.transform(Xte)).argmax(1) == yte).mean())

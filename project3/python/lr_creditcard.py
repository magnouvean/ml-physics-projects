import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

from generate_data import load_data_creditcard

np.random.seed(11235)

X_train, y_train, X_test, y_test = load_data_creditcard(val_split=False)

# We will try both l1 and l2 (lasso and ridge) type penalties
clf_l1 = LogisticRegressionCV(penalty="l1", solver="saga")
clf_l2 = LogisticRegressionCV(penalty="l2")

clf_l1.fit(X_train, y_train)
clf_l2.fit(X_train, y_train)

y_test_pred_l1 = clf_l1.predict(X_test)
y_test_pred_l2 = clf_l2.predict(X_test)

print("Logistic regression (CV) with L1 regularization")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_l1)}")
print("Logistic regression (CV) with L2 regularization")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_l2)}")

fig, ax = plt.subplots(1, 2)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred_l1, ax=ax[0], colorbar=False, values_format="d"
)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred_l2, ax=ax[1], colorbar=False, values_format="d"
)

ax[0].set_title("L1 regularization")
ax[1].set_title("L2 regularization")

fig.savefig("../figures/logistic_regression_confusion_mat.png")

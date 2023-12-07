import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from generate_data import load_data_creditcard

np.random.seed(11235)

X_train, y_train, X_test, y_test = load_data_creditcard(val_split=False)

# We will try both l1 and l2 (lasso and ridge) type penalties, and start
# without any polynomial features.
clf_l1 = LogisticRegressionCV(penalty="l1", solver="saga")
clf_l2 = LogisticRegressionCV(penalty="l2")

clf_l1.fit(X_train, y_train)
clf_l2.fit(X_train, y_train)

y_test_pred_l1_proba = clf_l1.predict_proba(X_test)[:, 1]
y_test_pred_l2_proba = clf_l2.predict_proba(X_test)[:, 1]
y_test_pred_l1 = clf_l1.predict(X_test)
y_test_pred_l2 = clf_l2.predict(X_test)

# Print the metrics for the two models
print("Logistic regression (CV) with L1 regularization")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_l1)}")
print("Logistic regression (CV) with L2 regularization")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_l2)}")

# Create confusion matrices as well
fig_cm, ax_cm = plt.subplots(1, 2)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred_l1, ax=ax_cm[0], colorbar=False, values_format="d"
)
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred_l2, ax=ax_cm[1], colorbar=False, values_format="d"
)

ax_cm[0].set_title("L1 regularization")
ax_cm[1].set_title("L2 regularization")

fig_cm.savefig("../figures/lr_confusion_mat.png", dpi=196)

# And then ROC-curves as well
fig_roc, ax_roc = plt.subplots(1, 2)
RocCurveDisplay.from_predictions(y_test, y_test_pred_l1_proba, ax=ax_roc[0])
RocCurveDisplay.from_predictions(y_test, y_test_pred_l2_proba, ax=ax_roc[1])

ax_roc[0].set_title("L1 regularization")
ax_roc[1].set_title("L2 regularization")
ax_roc[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.25))
ax_roc[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.25))

fig_roc.subplots_adjust(wspace=0.5)
fig_roc.tight_layout()
fig_roc.savefig("../figures/lr_roc_curve.png", dpi=196)


# We now try with polynomial features (and also scale the polynomial features)
poly = PolynomialFeatures(degree=2)
sc = StandardScaler()
X_train = sc.fit_transform(poly.fit_transform(X_train))
X_test = sc.transform(poly.transform(X_test))

# We only use l2 regularization now
clf_poly = LogisticRegressionCV(penalty="l2")
clf_poly.fit(X_train, y_train)

# Predict and give accuracy
y_test_pred_proba = clf_poly.predict_proba(X_test)[:, 1]
y_test_pred = clf_poly.predict(X_test)
print("Logistic regression (CV) with polynomial features")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred)}")

# Finally create confusion matrix for the polynomial feature model
ConfusionMatrixDisplay.from_predictions(
    y_test, y_test_pred, colorbar=False, values_format="d"
)
plt.savefig("../figures/lr_poly_confusion_mat.png", dpi=196)
# And a ROC-curve for our polynomial model
RocCurveDisplay.from_predictions(y_test, y_test_pred_proba)
plt.savefig("../figures/lr_poly_roc_curve.png", dpi=196)

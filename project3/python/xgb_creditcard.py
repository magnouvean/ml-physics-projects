import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

from generate_data import load_data_creditcard
from sklearn.metrics import ConfusionMatrixDisplay

X_train, y_train, X_val, y_val, X_test, y_test = load_data_creditcard()
y_train, y_val, y_test = (
    y_train.values.ravel(),
    y_val.values.ravel(),
    y_test.values.ravel(),
)
X_train_smaller, y_train_smaller = X_train[:10_000], y_train[:10_000]
X_val_smaller, y_val_smaller = X_val[:10_000], y_val[:10_000]

# We will here be testing with n_estimators varying from 10 at the lowest to
# 500 at the highest and check with 10. We can do this without it taking too
# much time due to the fact that these trees are quite quick to train.
n_estimators_params = np.arange(10, 510, 10)
accuracies = np.zeros(len(n_estimators_params))
for i, n_trees in enumerate(n_estimators_params):
    clf = xgb.XGBClassifier(n_estimators=n_trees)
    clf.fit(X_train_smaller, y_train_smaller)
    y_val_pred = clf.predict(X_val_smaller)
    accuracies[i] = np.mean(y_val_pred == y_val_smaller)

# Calculate the parameter of n_estimators, which gave the best results for the
# validation data and print this to the output.
best_n_estimators = n_estimators_params[np.argmax(accuracies)]
print(
    "====Best number of estimators====\n"
    f"{best_n_estimators}, accuracy: {np.max(accuracies)}"
)

max_leaves_params = np.arange(2, 12)
accuracies = np.zeros(len(max_leaves_params))
for i, max_leaves in enumerate(max_leaves_params):
    clf = xgb.XGBClassifier(n_estimators=best_n_estimators, max_leaves=max_leaves)
    clf.fit(X_train_smaller, y_train_smaller)
    y_val_pred = clf.predict(X_val_smaller)
    accuracies[i] = np.mean(y_val_pred == y_val_smaller)

# Determine the best max_leaves hyperparameter for the validation data and
# print this to the output.
best_max_leaves = max_leaves_params[np.argmax(accuracies)]
print(f"\n====Best max leaves====\n{best_max_leaves}, accuracy: {np.max(accuracies)}")

# We now are ready to train the final model. We merge the train and validation
# set together for the final model.
del X_train_smaller
del y_train_smaller
X_train = np.concatenate([X_train, X_val])
y_train = np.concatenate([y_train, y_val])

# Train final model.
clf = xgb.XGBClassifier(n_estimators=best_n_estimators, max_leaves=best_max_leaves)
clf.fit(X_train, y_train)
# And make predictions
y_test_pred = clf.predict(X_test)
final_accuracy = np.mean(y_test_pred == y_test)

# Give out the final accuracy.
print(f"\n====Final accuracy====\n{final_accuracy}")

# Also create a confusion matrix for the final predictions.
ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
plt.savefig("../figures/xgb_final_confusion_mat.png")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression as LogisticRegressionSK
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from p2libs import LogisticRegression, SchedulerAdam, fig_directory

# It is important to set the seed for reproducibility
np.random.seed(1234)

X, y = load_breast_cancer(return_X_y=True)

# Train-test-validation splitting
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.4)

# Scale the data as well
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)


lmbdas = 10 ** (np.linspace(-10, 2, 13))
accuracies_train = np.zeros(len(lmbdas))
accuracies_val = np.zeros_like(accuracies_train)
for i, lmbda in enumerate(lmbdas):
    scheduler = SchedulerAdam(0.01)
    model = LogisticRegression(scheduler, lmbda=lmbda)
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train, probs=False)
    y_pred_val = model.predict(X_val, probs=False)
    accuracies_train[i] = np.mean(y_pred_train == y_train)
    accuracies_val[i] = np.mean(y_pred_val == y_val)

print(
    f"Best accuracy (train): {np.max(accuracies_train)}, lambda={lmbdas[np.argmax(accuracies_train)]}"
)
print(
    f"Best accuracy (validation): {np.max(accuracies_val)}, lambda={lmbdas[np.argmax(accuracies_val)]}"
)

# We now just select the lambda based on the one which gave the best validation
# accuracy.
lmbda = lmbdas[np.argmax(accuracies_val)]

# And we then train the final model on this.
X_train = np.concatenate([X_train, X_val])
y_train = np.concatenate([y_train, y_val])

model = LogisticRegression(SchedulerAdam(0.05), lmbda=lmbda)
model.fit(X_train, y_train, epochs=2000)
y_pred_test = model.predict(X_test, probs=False)
final_accuracy = np.mean(y_pred_test == y_test)
print(f"\n\nFinal model accuracy: {final_accuracy}")
# Create confusion matrix for our own model.
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
disp.ax_.set_title("Confusion matrix for logistic regression")
plt.savefig(f"{fig_directory}/confusion_matrix_logreg.png")

# Now we compare this with sklearn to ensure that our implementation is
# correct.
model_skl = LogisticRegressionSK(
    C=1 / lmbda, penalty="l2", max_iter=2000, fit_intercept=False
)
model_skl.fit(X_train, y_train)
y_pred_test_skl = model_skl.predict(X_test)
final_accuracy_skl = np.mean(y_pred_test_skl == y_test)
print(f"Final model accuracy sklearn: {final_accuracy_skl}")
# Create confusion matrix for the sklearn model.
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
disp.ax_.set_title("Confusion matrix for logistic regression (sklearn)")
plt.savefig(f"{fig_directory}/confusion_matrix_logreg_skl.png")

print("\n\n====Sklearn vs own model betas====")
print("Own model:")
print(model._betas)
print("Sklearn model:")
print(model_skl.coef_)

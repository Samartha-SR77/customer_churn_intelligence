import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

df = pd.read_csv("data/telco_churn_cleaned.csv")

X = df.drop(columns=["Churn", "customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

xt, xs, yt, ys = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

x_train, x_val, y_train, y_val = train_test_split(
    xt,
    yt,
    test_size=0.20,
    random_state=42,
    stratify=yt
)

categorical_cols = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_cols = X.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ("num", StandardScaler(), numerical_cols)
])

baseline_lr = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

baseline_rf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1
    ))
])

rf_tuning = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        random_state=42,
        n_jobs=1
    ))
])

param_grid = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [10, 15, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
    "classifier__class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
    estimator=rf_tuning,
    param_grid=param_grid,
    scoring="recall",
    cv=5,
    n_jobs=1,
    verbose=1
)

grid_search.fit(x_train, y_train)

best_rf = grid_search.best_estimator_

print("\nBEST RANDOM FOREST PARAMETERS")
print(grid_search.best_params_)
print("Best CV Recall:", grid_search.best_score_)

baseline_lr.fit(x_train, y_train)
baseline_rf.fit(x_train, y_train)

models = {
    "Logistic Regression": baseline_lr,
    "Original Random Forest": baseline_rf,
    "Tuned Random Forest": best_rf
}

comparison = []

for name, estimator in models.items():
    val_pred = estimator.predict(x_val)
    val_prob = estimator.predict_proba(x_val)[:, 1]

    report = classification_report(
        y_val,
        val_pred,
        output_dict=True,
        zero_division=0
    )

    comparison.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, val_pred),
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1 Score": report["1"]["f1-score"],
        "ROC-AUC": roc_auc_score(y_val, val_prob)
    })

comparison_df = pd.DataFrame(comparison)

print("\nVALIDATION MODEL COMPARISON")
print(comparison_df.round(3).to_string(index=False))

val_prob = best_rf.predict_proba(x_val)[:, 1]

threshold_results = []

for threshold in [0.50, 0.45, 0.40, 0.35, 0.30]:
    val_pred = (val_prob >= threshold).astype(int)

    report = classification_report(
        y_val,
        val_pred,
        output_dict=True,
        zero_division=0
    )

    threshold_results.append({
        "Threshold": threshold,
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1 Score": report["1"]["f1-score"],
        "Accuracy": accuracy_score(y_val, val_pred)
    })

threshold_df = pd.DataFrame(threshold_results)

print("\nVALIDATION THRESHOLD COMPARISON")
print(threshold_df.round(3).to_string(index=False))

final_model = best_rf
final_threshold = 0.50

test_prob = final_model.predict_proba(xs)[:, 1]
test_pred = (test_prob >= final_threshold).astype(int)

test_report = classification_report(
    ys,
    test_pred,
    target_names=["Stayed", "Churned"],
    output_dict=True,
    zero_division=0
)

test_accuracy = accuracy_score(ys, test_pred)
test_roc_auc = roc_auc_score(ys, test_prob)

print("\nFINAL TEST EVALUATION")
print("Selected Threshold:", final_threshold)
print("Accuracy:", round(test_accuracy, 4))
print("ROC-AUC:", round(test_roc_auc, 4))

print(classification_report(
    ys,
    test_pred,
    target_names=["Stayed", "Churned"],
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(ys, test_pred))

feature_names = final_model.named_steps[
    "preprocessor"
].get_feature_names_out()

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": final_model.named_steps[
        "classifier"
    ].feature_importances_
}).sort_values("Importance", ascending=False)

print("\nTOP 15 IMPORTANT FEATURES")
print(feature_importance.head(15).to_string(index=False))

os.makedirs("models", exist_ok=True)

joblib.dump(final_model, "models/churn_model.joblib")

model_config = {
    "model_name": "Tuned Random Forest",
    "threshold": final_threshold,
    "best_parameters": grid_search.best_params_,
    "cv_recall": float(grid_search.best_score_),
    "test_accuracy": float(test_accuracy),
    "test_roc_auc": float(test_roc_auc),
    "test_churn_precision": float(test_report["Churned"]["precision"]),
    "test_churn_recall": float(test_report["Churned"]["recall"]),
    "test_churn_f1": float(test_report["Churned"]["f1-score"])
}

with open("models/model_config.json", "w") as file:
    json.dump(model_config, file, indent=4)

print("\nMODEL SAVED SUCCESSFULLY")
print("models/churn_model.joblib")
print("models/model_config.json")
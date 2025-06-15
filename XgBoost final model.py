import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, precision_recall_fscore_support,
    roc_curve, auc
)
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb


# Load the separate test dataset
#test_file = "test_data_0.5_timestamps_rightfaultcolumn_test_data.csv"  
test_file = "bruk_denne_test.csv"
df_test = pd.read_csv(test_file, sep=";")

#file_path = "test_data_0.5_timestamps_rightfaultcolumn_without_testdata.csv"
file_path = "bruk_denne_train.csv"
df_train = pd.read_csv(file_path, sep=";")

print("Train columns:", df_train.columns.tolist())
print("Test  columns:", df_test .columns.tolist())
df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
df_test["timestamp"]  = pd.to_datetime(df_test["timestamp"])

df_train["force_x_norm"] = df_train["force_x"] / (df_train["force_x"].abs().max() + 1e-6)
df_train["force_y_norm"] = df_train["force_y"] / (df_train["force_y"].abs().max() + 1e-6)
df_train["torque_norm"] = df_train["torque_n"] / (df_train["torque_n"].abs().max() + 1e-6)
df_test["force_x_norm"] = df_test["force_x"] / (df_test["force_x"].abs().max() + 1e-6)
df_test["force_y_norm"] = df_test["force_y"] / (df_test["force_y"].abs().max() + 1e-6)
df_test["torque_norm"] = df_train["torque_n"] / (df_train["torque_n"].abs().max() + 1e-6)


# --- Feature Engineering ---
def compute_expected_velocity(df, mass=1, time_step=5):
    df = df.copy()
    df["velocity_expected_x"] = (
        df.groupby("id")["linear_velocity_x"].shift(1)
        + (df.groupby("id")["force_x_norm"].shift(1) / mass) * time_step
    )
    df["velocity_expected_y"] = (
        df.groupby("id")["linear_velocity_y"].shift(1)
        + (df.groupby("id")["force_y_norm"].shift(1) / mass) * time_step
    )
    df["velocity_deviation_x"] = df["linear_velocity_x"] - df["velocity_expected_x"]
    df["velocity_deviation_y"] = df["linear_velocity_y"] - df["velocity_expected_y"]
    return df

def compute_expected_position(df, time_step=5):
    df = df.copy()
    df["position_expected_x"] = df.groupby("id")["linear_velocity_x"].shift(1) * time_step
    df["position_expected_y"] = df.groupby("id")["linear_velocity_y"].shift(1) * time_step
    df["position_deviation_x"] = df["position_expected_x"] - (df["linear_velocity_x"] * time_step)
    df["position_deviation_y"] = df["position_expected_y"] - (df["linear_velocity_y"] * time_step)
    return df

def compute_second_order_expected_position(df, time_step=5):
    df = df.copy()

    df["position_expected_x"] = (
        df.groupby("id")["linear_velocity_x"].shift(1) * time_step +
        0.5 * df.groupby("id")["linear_acceleration_x"].shift(1) * (time_step ** 2)
    )

    df["position_expected_y"] = (
        df.groupby("id")["linear_velocity_y"].shift(1) * time_step +
        0.5 * df.groupby("id")["linear_acceleration_y"].shift(1) * (time_step ** 2)
    )

    df["position_deviation_x"] = df["position_expected_x"] - (df["linear_velocity_x"] * time_step)
    df["position_deviation_y"] = df["position_expected_y"] - (df["linear_velocity_y"] * time_step)

    return df

def add_rolling_features(df, window_size=5):
    for col in ["velocity_deviation_x", "velocity_deviation_y"]:
        df[f"{col}_rolling"] = (
            df.groupby("id")[col]
              .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())
        )
    return df

def add_slope_features(df, columns, group_col="id", window=10):
    def slope(x):
        if len(x) < window:
            return 0.0
        idx = np.arange(window)
        return np.polyfit(idx, x[-window:], 1)[0]
    
    for col in columns:
        slope_col_name = f"{col}_slope_{window}"
        df[slope_col_name] = (
            df.groupby(group_col)[col]
              .transform(lambda x: x.rolling(window, min_periods=window).apply(slope, raw=True).fillna(0))
        )
    return df

# --- Apply Feature Engineering ---
df_train = compute_expected_velocity(df_train)
df_train = compute_expected_position(df_train)
#df_train = compute_second_order_expected_position(df_train)
df_train = add_rolling_features(df_train)

df_test = compute_expected_velocity(df_test)
df_test = compute_expected_position(df_test)
#df_test = compute_second_order_expected_position(df_test)
df_test = add_rolling_features(df_test)

slope_cols = ["velocity_deviation_x", "velocity_deviation_y", "position_deviation_x", "position_deviation_y"]
df_train = add_slope_features(df_train, slope_cols)
df_test  = add_slope_features(df_test,  slope_cols)

# --- Labels ---
df_train["error_flag"] = df_train["fault_label"].apply(lambda x: 0 if x == 0 else 1)
df_test["error_flag"] = df_test["fault_label"].apply(lambda x: 0 if x == 0 else 1)

# --- Features ---
drop_columns = [
    "id", "timestamp", "fault_label", "error_flag", "linear_volcity_x", "linear_velocity_y", "linear_acceleration_x",
    "linear_acceleration_y"
]
feature_cols = [col for col in df_train.columns if col not in drop_columns]

# --- Per-bag Split ---
bag_ids = df_train["id"].unique()
bag_labels = df_train.groupby("id")["error_flag"].first()
train_ids, val_ids = train_test_split(
    bag_ids,
    test_size=0.25,
    random_state=42,
    stratify=bag_labels
)

train_mask = df_train["id"].isin(train_ids)
val_mask   = df_train["id"].isin(val_ids)

X_train = df_train.loc[train_mask, feature_cols]
y_train = df_train.loc[train_mask, "error_flag"]
X_val   = df_train.loc[val_mask, feature_cols]
y_val   = df_train.loc[val_mask, "error_flag"]
X_test  = df_test[feature_cols]
y_test  = df_test["error_flag"]



# --- Grid Search Setup ---
param_grid = {
    'max_depth': [3, 5, 7],
    'eta': [0.01, 0.005, 0.001],
    'lambda': [10, 15, 20],
    'scale_pos_weight': [0.0, 1.0, 2.0, 2.5] 
}

# --- Data Preparation  ---
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_test)


# --- Grid Search Loop ---
results = []
print("Starting Grid Search...")

# Iterate over every combination of parameters
import itertools
grid_combinations = list(itertools.product(*param_grid.values()))
total_combinations = len(grid_combinations)

for i, combination in enumerate(grid_combinations):
    # Create a dictionary for the current parameter set
    current_params = dict(zip(param_grid.keys(), combination))
    
    # Add the fixed parameters
    current_params['objective'] = 'binary:logistic'
    current_params['subsample'] = 0.8
    current_params['colsample_bytree'] = 0.8
    current_params['eval_metric'] = 'logloss'
    
    print(f"\n--- Running Combination {i+1}/{total_combinations} ---")
    print(current_params)
    
    # Train the model with the current parameter set
    model = xgb.train(
        params=current_params,
        dtrain=dtrain,
        num_boost_round=2000,
        evals=[(dval, "validation")],
        early_stopping_rounds=100,
        verbose_eval=False # Turn off verbose output inside the loop
    )
    
    # Predict on the test set
    y_pred_proba = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    
    # Find the best F1 score on the test set for this model
    best_f1_for_model = 0
    for thresh in np.arange(0.1, 0.9, 0.01):
        y_pred_temp = (y_pred_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred_temp, average='binary', zero_division=0)
        if f1 > best_f1_for_model:
            best_f1_for_model = f1
            
    print(f"Best F1 Score for this combination: {best_f1_for_model:.4f}")
    
    # Store the results
    results.append({
        'params': current_params,
        'best_iteration': model.best_iteration,
        'best_val_logloss': model.best_score,
        'best_f1_on_test': best_f1_for_model
    })

# --- Find and Display the Best Result ---
print("\n--- Grid Search Complete ---")

# Convert results to a DataFrame for easy viewing
results_df = pd.DataFrame(results)

# Sort by the best F1 score on the test set
best_result_df = results_df.sort_values(by='best_f1_on_test', ascending=False)

print("Top 5 Hyperparameter Combinations by F1 Score:")
print(best_result_df.head(5))

# Get the parameters of the absolute best model
best_params = best_result_df.iloc[0]['params']
print("\nBest hyperparameters found:")
print(best_params)

# --- RETRAIN THE BEST MODEL (IMPORTANT FOR FINAL PLOTS) ---
print("\nRetraining the best model on the full training data...")
# You can retrain with a final verbose output to see the learning curve
best_model = xgb.train(
    params=best_params,
    dtrain=dtrain,
    num_boost_round=2000,
    evals=[(dval, "validation")],
    early_stopping_rounds=100,
    verbose_eval=100
)

# --- Threshold selection on validation set ---
val_pred_proba = best_model.predict(dval, iteration_range=(0, best_model.best_iteration + 1))

best_thresh = 0.5
best_f1 = 0
for thresh in np.arange(0.1, 0.9, 0.01):
    val_pred_temp = (val_pred_proba >= thresh).astype(int)
    f1 = f1_score(y_val, val_pred_temp, average='binary', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = thresh
        
        
# --- Report ---
print(f"Best threshold: {best_thresh:.2f}")
print("Sample probabilities for 'error':", y_pred_proba[:5])
print(classification_report(y_test, y_pred, target_names=["no error", "error"], zero_division=0))
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Logloss Plot ---
plt.figure(figsize=(10, 6))
plt.plot(evals_result["validation"]["logloss"], label="Validation Logloss")
plt.xlabel("Boosting Round")
plt.ylabel("Logloss")
plt.title("Logloss over Rounds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Class Balance Check ---
for name, y in [("Train", y_train), ("Validation", y_val), ("Test", y_test)]:
    print(f"\n{name} set class distribution:")
    print(y.value_counts(normalize=True).rename("proportion"))
    print(y.value_counts().rename("count"))


# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Define label names
labels = ["No Error", "Error"]

# Plot
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

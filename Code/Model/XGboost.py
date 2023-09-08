import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# 1. Loading training data from a file
# Initialize an empty list to store the data
data = []
# Open the file and read line by line
with open('Path_to_your_ML_flow_dataset.txt', 'r') as file:
    for line in file.readlines():
        # Split the line based on commas and convert to respective data types
        line = line.strip().split(',')
        features = list(map(float, line[2:]))
        label = int(line[1])
        data.append((features, label))

# Convert the list of tuples to separate arrays for features (X) and labels (y)
X, y = zip(*data)
X = np.array(X)
y = np.array(y)

# Loading testing data similarly as above
data_test = []
with open('Path_to_your_ML_test_flow_dataset.txt', 'r') as file:
    for line in file.readlines():
        line = line.strip().split(',')
        features = list(map(float, line[2:]))
        label = int(line[1])
        data_test.append((features, label))

X_test, y_test = zip(*data_test)
X_test = np.array(X_test)
y_test = np.array(y_test)

# 2. Normalize the data to be within a range of [0, 1]
# This helps the gradient-based methods converge faster
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# It's essential to use the same scaler object to transform the test data to ensure consistency
X_test = scaler.transform(X_test)

# 3. Data Augmentation: Introduce a simple form of data augmentation by adding noise to the data
# This can sometimes help improve model robustness
def augment_data(X, y, noise_level=0.01):
    noise = np.random.normal(0, noise_level, X.shape)
    X_aug = X + noise
    return np.vstack([X, X_aug]), np.hstack([y, y])

X, y = augment_data(X, y)

# 4. Train the XGBoost model using cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_model = None  # Placeholder for the best model during cross-validation
best_auc = 0  # Placeholder for the best AUC value

# Split the data into train/validation sets based on the stratified k-fold splits
for train_index, val_index in kf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Convert data to XGBoost's internal DMatrix format for efficiency
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Define training parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'booster': 'gbtree',
        'nthread': 4,
        'eta': 0.1,
        'max_depth': 10,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # Define datasets for evaluation during training
    evals = [(dtrain, 'train'), (dval, 'eval')]
    num_round = 1000  # Maximum number of boosting rounds

    # Train the XGBoost model
    bst = xgb.train(params, dtrain, num_round, evals, early_stopping_rounds=20, verbose_eval=10)

    # Validate the model on the validation data
    y_pred_val = bst.predict(dval, ntree_limit=bst.best_ntree_limit)
    roc_auc_val = roc_auc_score(y_val, y_pred_val)

    # If current model's AUC is better than previously found best, update the best model
    if roc_auc_val > best_auc:
        best_auc = roc_auc_val
        best_model = bst

# 5. Evaluate the best model on the test set
dtest = xgb.DMatrix(X_test, label=y_test)
y_pred_test = best_model.predict(dtest, ntree_limit=best_model.best_ntree_limit)

# Calculate ROC curve metrics
fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
roc_auc_test = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) on Test Set')
plt.legend(loc="lower right")
plt.show()

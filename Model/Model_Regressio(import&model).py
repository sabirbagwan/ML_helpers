########### Imports ##############

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, max_error, mean_absolute_error, median_absolute_error


import warnings
warnings.filterwarnings(action='ignore')


############# train-test-split ##############
X = data.drop('color', axis = 1)
y = data['color']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

############### ModeL ####################


models = {
    "Linear Regression": LinearRegression(),
    "Linear Regression (L2 Regularization)": Ridge(),
    "Linear Regression (L1 Regularization)": Lasso(),
    "Linear Regression (L1/L2 Regularization)": ElasticNet(),
    "K-Nearest Neighbors": KNeighborsRegressor(),
    "Neural Network": MLPRegressor(),
    "Support Vector Machine (Linear Kernel)": LinearSVR(),
    "Support Vector Machine (RBF Kernel)": SVR(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(),
    "LightGBM": LGBMRegressor(),
    "CatBoost": CatBoostRegressor(verbose=0)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(name + " trained.")
    
params = {
    "Linear Regression (L2 Regularization)": {"alpha": [0.01, 0.1, 1, 10]},
    "Linear Regression (L1 Regularization)": {"alpha": [0.01, 0.1, 1, 10]},
    "Linear Regression (L1/L2 Regularization)": {"alpha": [0.01, 0.1, 1, 10], "l1_ratio": [0.1, 0.5, 0.9]},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 9]},
    "Neural Network": {"hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)], "alpha": [0.01, 0.001]},
    "Support Vector Machine (Linear Kernel)": {"C": [0.1, 1, 10]},
    "Support Vector Machine (RBF Kernel)": {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]},
    "Decision Tree": {"max_depth": [3, 5, 7, 9]},
    "Random Forest": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7, 9]},
    "Gradient Boosting": {"learning_rate": [0.01, 0.1, 1], "max_depth": [3, 5, 7, 9]},
    "XGBoost": {"learning_rate": [0.01, 0.1, 1], "max_depth": [3, 5, 7, 9]},
    "LightGBM": {"learning_rate": [0.01, 0.1, 1], "max_depth": [3, 5, 7, 9]},
    "CatBoost": {"learning_rate": [0.01, 0.1, 1], "max_depth": [3, 5, 7, 9]}
}

best_params = {}

for name, model in models.items():
    if name in params:
#         print('\n')
        print("Tuning hyperparameters for " + name)
        param_grid = params[name]
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print("Best parameters:", grid_search.best_params_)
        models[name] = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_
    else:
        print("Skipping " + name + " as it does not have any hyperparameters to tune.")


results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    adj_r2 = 1 - (1 - r2) * (len(X_test) - 1) / (len(X_test) - X_test.shape[1] - 1)
    mae = mean_absolute_error(y_test, y_pred)
    evs = explained_variance_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    results.append([name, mse, r2, adj_r2, mae, evs, mape])

results_df = pd.DataFrame(results, columns=["Model", "MSE", "R2 score", "Adj. R2", "MAE", "Explained Variance Score", "MAPE"])
results_df = results_df.sort_values(by=["R2 score"], ascending=False)
results_df

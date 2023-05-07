models = {
    "Logistic Regression": LogisticRegression(),
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "Support Vector Machine (Linear Kernel)": SVC(kernel='linear', probability=True),
    "Support Vector Machine (RBF Kernel)": SVC(kernel='rbf', probability=True)
}

params = {
    "Logistic Regression": {"C": [0.1, 1, 10]},
    "Linear Discriminant Analysis": {"solver": ["svd", "lsqr"]},
    "Quadratic Discriminant Analysis": {},
    "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7, 9]},
    "Decision Tree": {"max_depth": [3, 5, 7, 9]},
    "Random Forest": {"n_estimators": [50, 100, 150], "max_depth": [3, 5, 7, 9]},
    "Gradient Boosting": {"learning_rate": [0.01, 0.1, 1], "max_depth": [3, 5, 7, 9]},
    "XGBoost": {"learning_rate": [0.01, 0.1, 1], "max_depth": [3, 5, 7, 9]},
    "LightGBM": {"learning_rate": [0.01, 0.1, 1], "max_depth": [3, 5, 7, 9]},
    "CatBoost": {"learning_rate": [0.01, 0.1, 1], "max_depth": [3, 5, 7, 9]},
    "Support Vector Machine (Linear Kernel)": {"C": [0.1, 1, 10]},
    "Support Vector Machine (RBF Kernel)": {"C": [0.1, 1, 10], "gamma": [0.01, 0.1, 1]}
}

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    results.append([name, accuracy, precision, recall, f1, roc_auc])

results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 score", "ROC AUC"])
results_df = results_df.sort_values(by=["Accuracy"], ascending=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the data
data = pd.read_csv("model/filtered_dataset.csv")

# Drop the 'specific.disorder' column
data.drop("specific.disorder", axis=1, inplace=True)

# Split features and target
y = data["healthy"]
X = data.drop("healthy", axis=1)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, 
                                                    test_size=0.2, random_state=42)

# Dimensionality reduction with PCA (keeping 95% of the variance)
pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the models and their hyperparameter grids
models = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {
            "C": [0.1, 1, 10, 100],
            "solver": ["lbfgs"],  # 'lbfgs' is compatible with L2 penalty
            "penalty": ["l2"]
        }
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Support Vector Machine": {
        "model": SVC(random_state=42),
        "params": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        }
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7, 9],
            "weights": ["uniform", "distance"]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "learning_rate": [0.01, 0.1, 1],
            "max_depth": [3, 5, 7]
        }
    }
}

# Dictionary to store the results
best_estimators = {}
results = {}

# Run GridSearchCV for each model
for model_name, config in models.items():
    print(f"Starting grid search for {model_name}...")
    grid_search = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        scoring="accuracy",
        cv=5,  # 5-fold cross-validation
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_estimators[model_name] = grid_search.best_estimator_
    y_pred = grid_search.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[model_name] = acc
    
    print(f"{model_name} best parameters: {grid_search.best_params_}")
    print(f"{model_name} Accuracy = {acc:.4f}\n")

# Visualize the model performance
plt.figure(figsize=(10, 6))
model_names = list(results.keys())
accuracy_scores = list(results.values())
sns.barplot(x=accuracy_scores, y=model_names)
plt.xlabel("Accuracy Score")
plt.title("Model Comparison with Hyperparameter Tuning")
plt.show()

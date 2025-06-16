import mlflow
import mlflow.artifacts
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Remove the tracking URI line to use local MLflow
# mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load Wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 10
n_estimators = 5

mlflow.autolog()

# mention experiment
mlflow.set_experiment("MLFlow_MLOps_1")

with mlflow.start_run():
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)

    # created a confusion metrics plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.xlabel("Actual")
    plt.ylabel("predicted")
    plt.title("confusion_matrics")

    # save plot
    plt.savefig("Confusion_metrics.png")

    # log artifact using mlflow
    mlflow.log_artifact(__file__)

    # set tags 
    mlflow.set_tags({"Author": 'Anand', "Project": "Wine Classification"})

    print(accuracy)
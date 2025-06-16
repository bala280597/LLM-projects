import pandas as pd
from  sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("http://127.0.0.1:5000")
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

params = { 'solver': 'newton-cg', 'max_iter': 1000, 'random_state': 8888}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
mlflow.set_experiment("MLflow - 2")

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("Training info", "Basic LR model for iris data")

    signature = infer_signature(X_train, lr.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,

    )
model_uri = 'runs:/07e9065f44a94717be8961a61664eac1/iris_model'
loaded_model = mlflow.pyfunc.load_model(model_uri)
predictions = loaded_model.predict(X_test)
iris_features_name = datasets.load_iris().feature_names
result = pd.DataFrame(X_test, columns=iris_features_name)
result['actual'] = y_test
result['prediction'] = predictions

print(result)

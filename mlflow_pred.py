import mlflow
import numpy as np

# set the MLflow tracking URI
mlflow.set_tracking_uri("http://mlflow-web:5000")

# load the model
model_uri = "models:/iris-classification/None"
model = mlflow.pyfunc.load_model(model_uri)

# create a sample input
sample_input = np.array([[10.1, 8.5, 150.4, 300.2]])

# predict using the model
prediction = model.predict(sample_input)

print(f"Sample Input: {sample_input}")
print(f"Prediction: {prediction}")
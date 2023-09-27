from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime,timedelta
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Function to perform linear regression and get metrics
def perform_linear_regression_and_get_metrics():
    # Load the Boston housing dataset
    boston = load_boston()
    X = boston.data
    y = boston.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate mean squared error and R-squared
    mse = mean_squared_error(y_test, y_pred)
    r_squared = r2_score(y_test, y_pred)

    # Print the results
    print("Mean Squared Error:", mse)
    print("R-squared:", r_squared)

# Define the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 9, 22),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('linear_regression_dag', default_args=default_args, description='Perform Linear Regression and Get Metrics', schedule_interval='@daily')

# Define the task to perform linear regression and get metrics
linear_regression_task = PythonOperator(
    task_id='perform_linear_regression',
    python_callable=perform_linear_regression_and_get_metrics,
    dag=dag,
)

# Set task dependencies
linear_regression_task

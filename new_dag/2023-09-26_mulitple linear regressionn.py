from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'John Doe',
    'depends_on_past': False,
    'start_date': datetime(2020, 1, 1)
}

dag = DAG('multiple_linear_regression_pipeline', default_args=default_args, schedule_interval=None)

# download link: https://www.kaggle.com/uciml/pima-indians-diabetes-database
def download_dataset():
    #Downloads the dataset from Kaggle
    # Code to download the dataset
    pass
def process_data():
    #Processes the data
    # Code to process the data
    pass
def split_data():
    #Splits the data into training and testing dataset
    # Code to split the data
    pass
def apply_model():
    
    #Apply the model
    # Code to apply the multiple linear regression model
    pass
def evaluate_model():
    #Evaluates the model
    # Code to evaluate the model
    pass

# Define tasks
download_task = PythonOperator(task_id='download_dataset', python_callable=download_dataset, dag=dag)

process_data_task = PythonOperator(task_id='process_data', python_callable=process_data, dag=dag)

split_data_task = PythonOperator(task_id='split_data', python_callable=split_data, dag=dag)

apply_model_task = PythonOperator(task_id='apply_model', python_callable=apply_model, dag=dag)

eval_task = PythonOperator(task_id='evaluate_model', python_callable=evaluate_model, dag=dag)

# Set task sequence
download_task >> process_data_task >> split_data_task >> apply_model_task >> eval_task
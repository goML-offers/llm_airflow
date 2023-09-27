from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# default Arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2020,1,1),
    'retries':3,
    'retry_delay':timedelta(minutes=15),
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
}

# Define the DAG
with DAG(
    'Linear_Regression',
    default_args=default_args,
    schedule_interval="0 3 * * * ",
    catchup=False
) as dag:
    
    # Define Task1
    Task1_create_train_test_split_data = PythonOperator(
        task_id='create_train_test_split_data',
        python_callable=train_test_split_data,
        provide_context=True,
    )

    # Define Task2
    Task2_preprocess_data = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
        provide_context=True,
        dag=dag
    )    
    
    # Define Task3
    Task3_fit_data_into_model = PythonOperator(
        task_id='fit_data_into_model',
        python_callable=fit_data_into_model,
        provide_context=True,
        dag=dag
    )
    
    # Define Task4
    Task4_evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
        provide_context=True,
        dag=dag                                 
    )
    
    # Define Dependency Order
    Task1_create_train_test_split_data \
    >> Task2_preprocess_data \
    >> Task3_fit_data_into_model \
    >> Task4_evaluate_model
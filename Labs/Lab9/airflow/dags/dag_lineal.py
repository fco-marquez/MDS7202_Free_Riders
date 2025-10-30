import datetime as dt

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from hiring_functions import (
    create_folders,
    gradio_interface,
    predict,
    preprocess_and_train,
    split_data,
)

default_args = {
    "owner": "free-riders",
    "retries": 1,
}

with DAG(
    dag_id="hiring_lineal",
    default_args=default_args,
    description="Hiring Decision Workflow with Linear DAG",
    start_date=dt.datetime(2024, 10, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    # Task 1 - Just a simple print statement
    dummy_task = EmptyOperator(task_id="Starting_the_process", retries=2)

    # Task 2 - Create necessary folders
    create_folders_task = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"ds": "{{ ds }}"},
    )

    # Task 3 - Download data using BashOperator
    task_download_dataset_1 = BashOperator(
        task_id="download_dataset_1",
        bash_command="curl -o "
        "/root/airflow/{{ ds }}/raw/data_1.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv",
    )

    # Task 4 - Split data
    split_data_task = PythonOperator(
        task_id="split_data", python_callable=split_data, op_kwargs={"ds": "{{ ds }}"}
    )

    # Task 5 - Preprocess and train model
    preprocess_and_train_task = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
        op_kwargs={"ds": "{{ ds }}"},
    )

    # Task 6 - Set up Gradio interface
    gradio_interface_task = PythonOperator(
        task_id="gradio_interface",
        python_callable=gradio_interface,
        op_kwargs={"ds": "{{ ds }}"},
    )

    # Define task dependencies
    (
        dummy_task
        >> create_folders_task
        >> task_download_dataset_1
        >> split_data_task
        >> preprocess_and_train_task
        >> gradio_interface_task
    )

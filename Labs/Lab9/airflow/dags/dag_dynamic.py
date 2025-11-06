import datetime as dt

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.dates import days_ago
from hiring_dynamic_functions import (
    create_folders,
    evaluate_models,
    load_and_merge,
    split_data,
    train_model,
)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

default_args = {
    "owner": "free-riders",
    "retries": 1,
}

with DAG(
    dag_id="hiring_dynamic",
    default_args=default_args,
    description="Hiring Decision Workflow with Dynamic Tasks",
    start_date=dt.datetime(2024, 10, 1),
    schedule_interval="0 15 5 * *",  # cada dia 5 del mes a las 15:00 UCT
    catchup=True,
) as dag:

    # Task 1 - Inicial dummy task
    dummy_task = EmptyOperator(task_id="Starting_the_process", retries=2)

    # Task 2 - Crear las carpetas necesarias
    create_folders_task = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        op_kwargs={"ds": "{{ ds }}"},
    )

    # Branching: Antes del 1-11-2024 descargar solo data_1.csv, despues ambas
    branch_task = BranchPythonOperator(
        task_id="branch_task",
        python_callable=lambda **kwargs: (
            "download_dataset_1"
            if kwargs['execution_date'] < dt.datetime(2024, 11, 1, tzinfo=kwargs['execution_date'].tzinfo)
            else ["download_dataset_1", "download_dataset_2"]
        ),
        provide_context=True,
    )

    # Task 3a - Descargar dataset 1
    task_download_dataset_1 = BashOperator(
        task_id="download_dataset_1",
        bash_command="curl -o "
        "/root/airflow/{{ ds }}/raw/data_1.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv",
    )

    # Task 3b - Descargar dataset 2
    task_download_dataset_2 = BashOperator(
        task_id="download_dataset_2",
        bash_command="curl -o "
        "/root/airflow/{{ ds }}/raw/data_2.csv "
        "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv",
    )

    # Task 4 - Load and merge data con Trigger  para ejecutar si encuentra como minimo un dataset
    load_and_merge_task = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,
        op_kwargs={"ds": "{{ ds }}"},
        trigger_rule="one_success",
    )

    # Task 5 - Split data
    split_data_task = PythonOperator(
        task_id="split_data", python_callable=split_data, op_kwargs={"ds": "{{ ds }}"}
    )

    # Task 6 - Train models in parallel
    models = [
        RandomForestClassifier(random_state=1892, n_estimators=100),
        GradientBoostingClassifier(random_state=1892),
        LogisticRegression(max_iter=200),
    ]
    train_model_tasks = []
    for i, model in enumerate(models):
        task = PythonOperator(
            task_id=f"train_model_{i+1}",
            python_callable=train_model,
            op_kwargs={"ds": "{{ ds }}", "model": model},
        )
        train_model_tasks.append(task)

    # Task 7 - Evaluate models
    evaluate_models_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        op_kwargs={"ds": "{{ ds }}"},
        trigger_rule="all_success",
    )

    # Setting up dependencies
    (
        dummy_task
        >> create_folders_task
        >> branch_task
        >> [task_download_dataset_1, task_download_dataset_2]
        >> load_and_merge_task
        >> split_data_task
        >> train_model_tasks
        >> evaluate_models_task
    )

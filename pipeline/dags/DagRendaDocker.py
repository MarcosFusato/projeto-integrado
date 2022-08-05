from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.dummy import DummyOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.task_group import TaskGroup

pathScript = "/home/marcos/airflow/dags/scripts"
pathRenda = "/home/marcos/airflow/dags/scripts/arquivo/adultProcess.csv"
pathDocker = "/home/marcos/airflow/dags/scripts/apiRenda"

default_args = {
    'owner': 'marcos',
    'depends_on_past': False,
    'start_date': datetime(2019, 1, 1),
    'retries': 0,
}

with DAG(
    'dag-pipeline-renda-docker',
    schedule_interval=timedelta(minutes=10),
    catchup=False,
    default_args=default_args
) as dag:

    start = DummyOperator(task_id="start")

    t1 = BashOperator(
        task_id='etl_processamento',
        bash_command="""
       cd {0}
       python3 etl_processamento.py {1}
       """.format(pathScript, pathRenda)
    )

    t2 = BashOperator(
        task_id='modelo_sklearn',
        bash_command="""
       cd {0}
       python3 modelo_sklearn.py {1}
       """.format(pathScript, pathRenda)
    )

    t3 = BashOperator(
        task_id='validacao_modelo',
        bash_command="""
       cd {0}
       python3 validacao_modelo.py
       """.format(pathScript)
    )

    t4 = BashOperator(
        task_id='deploy_docker',
        bash_command="""
         cd {0}
         docker build  -t flask-renda .
         docker run -d -p 5500:5500 flask-renda 
         """.format(pathDocker)
    )

    end = DummyOperator(task_id='end')
    start >> t1 >> t2 >> t3 >> t4 >> end

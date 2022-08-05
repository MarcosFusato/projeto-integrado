from mlflow.tracking import MlflowClient
import mlflow
import shutil

try:
    idExperiment = mlflow.create_experiment('Renda-classificação')
except:
    idExperiment = mlflow.get_experiment_by_name(
        'Renda-classificação').experiment_id

client = MlflowClient()
listModel = client.list_run_infos(idExperiment)

modelo = ""
metrica = 0
for model in listModel:
    print(model.artifact_uri, model.run_uuid)
    data = client.get_run(model.run_uuid).data
    if data.metrics['accuracy_score'] > metrica:
        metrica = data.metrics['accuracy_score']
        modelo = model.artifact_uri

shutil.copy(modelo[7:] + "/model/model.pkl", 'apiRenda/model.pkl')

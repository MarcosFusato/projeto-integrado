# Esse arquivo vai transformar os dados, criar as variaveis x, y, base balanceada e gerar o modelo
#outputPath = "D:/Marcos/Documento/Pos Graduacao/projeto-integrado/Pipeline/scripts/arquivo/adultProcess.csv"
#Ler o arquivo acima

import pandas as pd
from urllib.parse import urlparse
import argparse

# Biblioteca para PreparacaoDados para o modelo
from lib.preparacao_dados import PreparacaoDados as prep
from lib.experimento_mlflow import ExperimentoMlflow

# Biblioteca para treino, metricas 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Algoritmos sklearn
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

#Bibliotecas Mlflow
import mlflow
import mlflow.sklearn

parser = argparse.ArgumentParser()
parser.add_argument("inputPath", help="arquivo input", type=str)
args = parser.parse_args()

# cabeçalho com o nome das colunas do dataset
header_columns = ['Idade', 'Classe Trabalho', 'Peso Final', 'Educacao', 'Num Educacao', 'Estado Civil', 
                  'Ocupacao','Relacao', 'Raca', 'Sexo', 'Ganho Capital', 'Perda Capital', 'Horas Semana', 
                  'Pais Nativo', 'Renda']

# Faz a leitura dos dados com os devidos tratamentos
renda = pd.read_csv(args.inputPath, names=header_columns, skiprows=1)

# Lista de colunas para a criação do data frame
lista_colunas_aux = ['Idade', 'Ganho Capital', 'Perda Capital', 'Relacao', 'Raca','Sexo', 'Renda']

# Lista de colunas para transforma variaveis categorica em numérica
lista_colunas_dummy = ['Relacao', 'Raca','Sexo']

# Lista de colunas para deixar na mesma escala
lista_colunas_scale = ['Idade', 'Ganho Capital', 'Perda Capital']

# Retorna os valores de X, y
X, y = prep.retorna_data_x_y(lista_colunas_dummy, lista_colunas_scale, renda[lista_colunas_aux])

# Balanceamento da base de dados
X_balanc, y_balanc = prep.retorna_data_x_y_balanceada(X, y)

# Seta o experimento
try:
    idExperiment = mlflow.create_experiment('Renda-classificação')
except:
    idExperiment = mlflow.get_experiment_by_name('Renda-classificação').experiment_id

exp = ExperimentoMlflow()

# Separa em treino, teste e faz a predicao algoritmo XGB
X_trainXGB, X_testXGB, y_trainXGB, y_testXGB, modeloXGB, predicaoXGB = exp.executa_predicao_modelo('XGB', X_balanc, y_balanc)

# Grava no mlflow experimento do algoritmo XGB
with mlflow.start_run(experiment_id=idExperiment):
     mlflow.log_param("learning_rate", 0.1)
     mlflow.log_param("max_depth", 9)
     mlflow.log_param("n_estimators", 180)
     mlflow.log_param("nthread", 4)
     mlflow.log_param("seed", 42)     

     # Registra as metricas
     mlflow.log_metric("accuracy_score", accuracy_score(y_testXGB, predicaoXGB))
     mlflow.log_metric("precision_score", precision_score(y_testXGB, predicaoXGB))
     mlflow.log_metric("recall_score", recall_score(y_testXGB, predicaoXGB))
     mlflow.log_metric("f1_score", f1_score(y_testXGB, predicaoXGB))     

     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme     

     if tracking_url_type_store != "file":
         mlflow.sklearn.log_model(modeloXGB, "model", registered_model_name='ModeloXGBRenda')         
     else:
         mlflow.sklearn.log_model(modeloXGB, "model")         

# Separa em treino, teste e faz a predicao algoritmo Random Forest
X_trainRF, X_testRF, y_trainRF, y_testRF, modeloRF, predicaoRF = exp.executa_predicao_modelo('RF', X_balanc, y_balanc)

# Grava no mlflow experimento do algoritmo RF
with mlflow.start_run(experiment_id=idExperiment):
     mlflow.log_param("bootstrap", 'True')
     mlflow.log_param("max_depth", 70)
     mlflow.log_param("max_features", 3)
     mlflow.log_param("min_samples_leaf", 3)
     mlflow.log_param("min_samples_split", 4)
     mlflow.log_param("n_estimators", 700)
     mlflow.log_param("random_state", 1)     

     # Registra as metricas
     mlflow.log_metric("accuracy_score", accuracy_score(y_testRF, predicaoRF))
     mlflow.log_metric("precision_score", precision_score(y_testRF, predicaoRF))
     mlflow.log_metric("recall_score", recall_score(y_testRF, predicaoRF))
     mlflow.log_metric("f1_score", f1_score(y_testRF, predicaoRF))     

     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme     
     
     if tracking_url_type_store != "file":
         mlflow.sklearn.log_model(modeloRF, "model", registered_model_name='ModeloRFRenda')         
     else:
         mlflow.sklearn.log_model(modeloRF, "model")         

# Separa em treino, teste e faz a predicao algoritmo Decision Tree
X_trainDT, X_testDT, y_trainDT, y_testDT, modeloDT, predicaoDT = exp.executa_predicao_modelo('DT', X_balanc, y_balanc)
                                                                    
# Grava no mlflow experimento do algoritmo Decison Tree
with mlflow.start_run(experiment_id=idExperiment):
     mlflow.log_param("criterion", 'entropy')
     mlflow.log_param("max_depth", 14)     
     mlflow.log_param("random_state", 1)

     # Registra as metricas
     mlflow.log_metric("accuracy_score", accuracy_score(y_testDT, predicaoDT))
     mlflow.log_metric("precision_score", precision_score(y_testDT, predicaoDT))
     mlflow.log_metric("recall_score", recall_score(y_testDT, predicaoDT))
     mlflow.log_metric("f1_score", f1_score(y_testDT, predicaoDT))

     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme    
     
     if tracking_url_type_store != "file":
         mlflow.sklearn.log_model(modeloDT, "model", registered_model_name='ModeloDTRenda')         
     else:
         mlflow.sklearn.log_model(modeloDT, "model")         


# Separa em treino, teste e faz a predicao algoritmo SVM
X_trainSVM, X_testSVM, y_trainSVM, y_testSVM, modeloSVM, predicaoSVM = exp.executa_predicao_modelo('SVM', X_balanc, y_balanc)
                                                                    
# Grava no mlflow experimento do algoritmo SVM
with mlflow.start_run(experiment_id=idExperiment):
     mlflow.log_param("c", 100)
     mlflow.log_param("gamma", 1)     
     mlflow.log_param("kernel", 'poly')     
     mlflow.log_param("random_state", 1)

     # Registra as metricas
     mlflow.log_metric("accuracy_score", accuracy_score(y_testSVM, predicaoSVM))
     mlflow.log_metric("precision_score", precision_score(y_testSVM, predicaoSVM))
     mlflow.log_metric("recall_score", recall_score(y_testSVM, predicaoSVM))
     mlflow.log_metric("f1_score", f1_score(y_testSVM, predicaoSVM))

     tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme    
     
     if tracking_url_type_store != "file":
         mlflow.sklearn.log_model(modeloSVM, "model", registered_model_name='ModeloDTRenda')         
     else:
         mlflow.sklearn.log_model(modeloSVM, "model")
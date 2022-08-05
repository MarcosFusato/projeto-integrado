import pandas as pd

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


class ExperimentoMlflow:
  
    def retorna_modelo(self, tipo):
        if tipo == 'XGB':
            modelo = XGBClassifier(objective='binary:logistic', 
                                    learning_rate=0.1, 
                                    max_depth=9, 
                                    n_estimators=180,
                                    nthread=4, 
                                    seed=42) 
        elif tipo == 'RF':
            modelo = RandomForestClassifier(bootstrap=True, 
                                            max_depth=70, 
                                            max_features=2, 
                                            min_samples_leaf=3, 
                                            min_samples_split=4, 
                                            n_estimators=700,
                                            random_state=1)
        elif tipo == 'DT':
            modelo = DecisionTreeClassifier(criterion='entropy', 
                                            max_depth=14,
                                            random_state=1)
        else:
            modelo = SVC(C=100, 
                        gamma=1,
                        kernel='poly',
                        random_state=1)

        return modelo


    def executa_predicao_modelo(self, tipo, X, y):
        # Separa base de treino e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

        # Cria uma instancia do algoritmo
        modelo = self.retorna_modelo(tipo)

        # Treina o modelo
        if tipo == 'XGB':
            modelo.fit(X_train, y_train, eval_metric='rmse')
        else:
            modelo.fit(X_train, y_train)

        # Faz a predição
        predicao = modelo.predict(X_test)

        # Retorna as variaveis
        return X_train, X_test, y_train, y_test, modelo, predicao
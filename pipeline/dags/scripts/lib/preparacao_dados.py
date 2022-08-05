from attr import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN

class PreparacaoDados:
                                
    def retorna_data_x_y(lista_colunas_dummy, lista_colunas_scale, data):

        # Transforma variaveis categoricas em numericas usando a função get_dummies do pandas
        dummy_features = pd.get_dummies(data[lista_colunas_dummy])

        # definindo a classe MinMaxScaler
        scaler = MinMaxScaler()
        
        # Aplicando o algoritmo para deixar os valores na mesma escala de 0 a 1
        data[lista_colunas_scale] = scaler.fit_transform(data[lista_colunas_scale])

        # Dataframe X
        data_x_aux = pd.concat([data, dummy_features], axis = 1)
        
        # Eliminando as variaveis após a conversão dos dados
        data_x_aux.drop(['Relacao', 'Raca', 'Sexo', 'Renda'], axis=1, inplace=True)
        
        # Cria dicionario para conversão dos dados
        dic_renda = {'<=50K': 0,'>50K': 1}
        
        # Cria data frame com o resultados da acuracia dos algoritmos
        data_y_aux = pd.DataFrame(columns = ['Renda'])

        # Dataframe y
        data_y_aux['Renda'] = data['Renda'].map(dic_renda)

        # Retorna os dataframes X e y
        return data_x_aux, data_y_aux         

    def retorna_data_x_y_balanceada(x, y):

        # definindo o algoritmo de balanceamento smoteen 
        smote_enn = SMOTEENN(random_state=0)

        # Faz o balanceamento da base de dados
        X_resampled, y_resampled = smote_enn.fit_resample(x, y)

        # Retorna as variaveis x e y
        return X_resampled, y_resampled



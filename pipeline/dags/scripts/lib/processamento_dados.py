import pandas as pd
import numpy as np

class ProcessamentoDados:

    def renomear_colunas (data):    
        # Cria um dicionario com o nome das colunas
        dic_colunas = {'age':'Idade', 'workclass':'Classe Trabalho', 'fnlwgt':'Peso Final', 'education':'Educacao', 
                       'education.num':'Num Educacao', 'marital.status':'Estado Civil','occupation':'Ocupacao',
                       'relationship':'Relacao', 'race':'Raca', 'sex':'Sexo', 'capital.gain':'Ganho Capital',
                       'capital.loss':'Perda Capital', 'hours.per.week':'Horas Semana', 'native.country': 'Pais Nativo',
                       'income': 'Renda'}
        print('dic_colunas: ', dic_colunas)
        
        #Renomeando as colunas baseado no dicionario
        data.rename(dic_colunas, axis=1, inplace=True) 
        
        return data    

    def remover_outlier_metodo_estatistico(data, coluna):
        #Calcula a Media
        valor_media = np.mean(data[coluna])      

        #Calcula o desvio padrão
        valor_desvio = np.std(data[coluna])  

        #Valor Corte
        valor_corte = valor_desvio * 2   

        #Valor de Ponto Minimo
        valor_ponto_minimo = valor_media - valor_corte 

        #Valor de Ponto Maximo
        valor_ponto_maximo = valor_media + valor_corte    

        #Indices a serem excluidos    
        idx = data[(data[coluna] < valor_ponto_minimo) | (data[coluna] > valor_ponto_maximo)].index     

        # removendo os dados da coluna baseado no indice
        data.drop(idx , inplace=True)  

        #Redifinindo os indices após a exclusão
        data.reset_index(inplace=True, drop=True)
        
        return data        
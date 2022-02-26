import pandas as pd
import numpy as np

class TratamentoDados:

    def tratamento_espaco_branco(data):    
        #Elimina os espaços em branco e mantem apenas "?"
        data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
        
        #Substitui a "?" por np.nan
        data = data.replace("?",np.nan)
        
        return data      

    def tratamento_dados_ausentes(data, nome_coluna):
        # Identifica a moda para a variavel nome_coluna
        var_nome_coluna = data[nome_coluna].mode()[0]

        # Substitui os dados faltantes dessa coluna pela moda
        data[nome_coluna].fillna(var_nome_coluna, inplace=True)       
        
        return data          
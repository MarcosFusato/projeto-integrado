import pandas as pd
import argparse

# Bibliotecas de Processamento e Tratamento dos dados
from lib.processamento_dados import ProcessamentoDados as pr
from lib.tratamento_dados import TratamentoDados as tr

parser = argparse.ArgumentParser()
parser.add_argument("outputPath", help="arquivo output", type=str)
args = parser.parse_args()
#outputPath = "D:/Marcos/Documento/Pos Graduacao/projeto-integrado/Pipeline/scripts/arquivo/adultProcess.csv"

# cabeçalho com o nome das colunas do dataset
header_columns = ['Idade', 'Classe Trabalho', 'Peso Final', 'Educacao', 'Num Educacao', 'Estado Civil', 
                  'Ocupacao','Relacao', 'Raca', 'Sexo', 'Ganho Capital', 'Perda Capital', 'Horas Semana', 
                  'Pais Nativo', 'Renda']

renda = pd.read_csv("base/adult.csv", names=header_columns, skiprows=1)

# Substitui os espaços em branco
renda = tr.tratamento_espaco_branco(renda)

# Tratamento de dados Ausentes
renda = tr.tratamento_dados_ausentes(renda, 'Classe Trabalho')
renda = tr.tratamento_dados_ausentes(renda, 'Ocupacao')
renda = tr.tratamento_dados_ausentes(renda, 'Pais Nativo')

# Tratamento de outliers
renda = pr.remover_outlier_metodo_estatistico(renda,'Idade')
renda = pr.remover_outlier_metodo_estatistico(renda,'Horas Semana')
renda = pr.remover_outlier_metodo_estatistico(renda,'Ganho Capital')

# Gera um arquivo csv com os dados já convertidos
renda.to_csv(args.outputPath, index=False)

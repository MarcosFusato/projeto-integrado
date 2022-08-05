from flask_restplus import Resource, Namespace, fields
from flask import jsonify, request
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn import preprocessing

api = Namespace('Predicao', description='Predição de Renda')

# definição de modelo que será validado ao receber post
modelo = api.model('PredicaoModel', {
    'idade': fields.Float,
    'ganho_capital': fields.Float,
    'perda_capital': fields.Float,
    'relacao_husband': fields.Float,
    'relacao_not_in_family': fields.Float,
    'relacao_other_relative': fields.Float,
    'relacao_own_child': fields.Float,
    'relacao_unmarried': fields.Float,
    'relacao_wife': fields.Float,
    'raca_amer_indian_eskimo': fields.Float,
    'raca_asian_pac_islander': fields.Float,
    'raca_black': fields.Float,
    'raca_other': fields.Float,
    'raca_white': fields.Float,
    'sexo_female': fields.Float,
    'sexo_male': fields.Float
})

with open("model.pkl", "rb") as arquivo:
    predictModel = pickle.load(arquivo)


@api.route('/')
class PredicaoController(Resource):

    @api.expect(modelo)  # espera modelo ao criar nova pessoa
    def post(self):

        # Recebe os dados do json
        data_json = request.json

        lista_colunas = ['Idade', 'Ganho Capital', 'Perda Capital', 'Relacao_Husband', 'Relacao_Not-in-family',
                         'Relacao_Other-relative', 'Relacao_Own-child', 'Relacao_Unmarried',
                         'Relacao_Wife', 'Raca_Amer-Indian-Eskimo', 'Raca_Asian-Pac-Islander',
                         'Raca_Black', 'Raca_Other', 'Raca_White', 'Sexo_Female', 'Sexo_Male']

        # Cria um dataframe
        data = pd.DataFrame(columns=lista_colunas)

        # Adiciona uma nova linha ao dataframe
        data = data.append({'Idade':  float(data_json['idade']),
                            'Ganho Capital': float(data_json['ganho_capital']),
                            'Perda Capital': float(data_json['perda_capital']),
                            'Relacao_Husband': float(data_json['relacao_husband']),
                            'Relacao_Not-in-family': float(data_json['relacao_not_in_family']),
                            'Relacao_Other-relative': float(data_json['relacao_other_relative']),
                            'Relacao_Own-child': float(data_json['relacao_own_child']),
                            'Relacao_Unmarried': float(data_json['relacao_unmarried']),
                            'Relacao_Wife': float(data_json['relacao_wife']),
                            'Raca_Amer-Indian-Eskimo': float(data_json['raca_amer_indian_eskimo']),
                            'Raca_Asian-Pac-Islander': float(data_json['raca_asian_pac_islander']),
                            'Raca_Black': float(data_json['raca_black']),
                            'Raca_Other': float(data_json['raca_other']),
                            'Raca_White': float(data_json['raca_white']),
                            'Sexo_Female': float(data_json['sexo_female']),
                            'Sexo_Male': float(data_json['sexo_male'])},
                           ignore_index=True)

        # definindo a classe MinMaxScaler
        scaler = MinMaxScaler()

        lista_colunas_scale = ['Idade', 'Ganho Capital', 'Perda Capital']

        # Aplicando o algoritmo para deixar os valores na mesma escala de 0 a 1
        data[lista_colunas_scale] = scaler.fit_transform(
            data[lista_colunas_scale])

        data = data.apply(pd.to_numeric)

        # Realiza a predição do modelo
        predicao = predictModel.predict(data)

        return jsonify({"Predict": str(predicao[0])})

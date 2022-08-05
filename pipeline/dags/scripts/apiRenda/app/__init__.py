from flask import Flask, Blueprint
from flask_restplus import Api
from werkzeug.contrib.fixers import ProxyFix

from app.main.predicao.predicao_controller import api as home_ns

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)
blueprint = Blueprint('api', __name__)
app.register_blueprint(blueprint)


api = Api(app, title='Api Flask Predição de Renda', version='1.0',
          description='Api de experimentos de previsão de renda com python flask', prefix='/api')

# adicionado namespace pessoa para rotas
api.add_namespace(home_ns, path='/predicao')

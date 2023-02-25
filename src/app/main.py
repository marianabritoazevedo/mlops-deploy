from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
import pickle
import os

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression

#Parte da predição de casas que só precisa ser executado uma vez -> Retirar, agora será lido a partir de arquivo pickle
# df = pd.read_csv('casas.csv')
colunas = ['tamanho', 'ano', 'garagem']
# X = df.drop('preco', axis=1)
# y = df['preco']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# modelo = LinearRegression()
# modelo.fit(X_train, y_train)

modelo = pickle.load(open('../../models/modelo.sav', 'rb'))

#Criação do app
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get('BASIC_AUTH_USERNAME')
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('BASIC_AUTH_PASSWORD')

basic_auth = BasicAuth(app)


#Definindo as rotas da API
@app.route('/')
def home():
    return "Minha primeira API"

#Recebe frase do usuário
@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(from_lang='pt_br', to='en')
    polaridade = tb_en.sentiment.polarity
    return "Polaridade: {}".format(polaridade)
    
#Predição de preco da casa
@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

#Executar aplicação
app.run(debug=True, host='0.0.0.0')
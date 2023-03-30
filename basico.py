# ---------------------------------------------------------------------------------------------------------------------------------------

# Certamente! Aqui estão alguns conceitos básicos de Python com exemplos:

# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                               VARIAVEIS 

# ---------------------------------------------------------------------------------------------------------------------------------------


# Variáveis: variáveis são usadas para armazenar dados em Python. Você pode criar uma variável atribuindo um valor a ela usando o operador "=".

# Exemplo:

x = 5
y = "Olá"


#                                                                ----------------------


# Variáveis de classe:
# Variáveis de classe são definidas dentro de uma classe, mas for a de qualquer método. 
# Eles são compartilhados por todas as instâncias da classe. Um exemplo de variável de classe pode ser um contador que 
# mantém o número de instâncias criadas a partir da classe.
 
# EXEMPLO: 

class Pessoa:
    # variável de classe
    count = 0
    def __init__(self, nome):
        self.nome = nome
        Pessoa.count += 1

p1 = Pessoa("João")
p2 = Pessoa("Maria")
print("Total de pessoas: ", Pessoa.count) # Total de pessoas: 2


#                                                                ----------------------


# Variáveis de instância:
# Variáveis de instância são definidas dentro de um método e pertencem apenas à instância atual da classe. 
# Cada instância terá seu próprio conjunto de valores para essas variáveis. Um exemplo pode ser um nome de pessoa ou um endereço de e-mail.

# EXEMPLO:

class Pessoa:
    def __init__(self, nome, idade):
        self.nome = nome # variável de instância
        self.idade = idade # variável de instância

p1 = Pessoa("João", 30)
p2 = Pessoa("Maria", 25)
print(p1.nome, p1.idade) # João 30
print(p2.nome, p2.idade) # Maria 25


#                                                                ----------------------


# Variáveis locais:
# Variáveis locais são definidas dentro de uma função e só podem ser acessadas dentro da função. 
# Elas são excluídas da memória assim que a função é concluída. Um exemplo pode ser uma variável que contém 
# o resultado de um cálculo temporário.

# EXEMPLO:

def calcular_idade(ano_nascimento):
    # variável local
    ano_atual = 2023 
    idade = ano_atual - ano_nascimento
    return idade

print(calcular_idade(1990)) # 33

#                                                                ----------------------

# Variáveis globais:
#  Variáveis globais são definidas for a de qualquer função ou classe e podem ser acessadas de qualquer lugar do programa. 
# No entanto, é uma prática recomendada evitá-las, pois podem causar efeitos colaterais inesperados em funções e classes. 
# Um exemplo pode ser uma variável que contém as configurações do programa.

TAMANHO_PAGINA = 10 # variável global

def listar_clientes(pagina):
    inicio = (pagina - 1) * TAMANHO_PAGINA
    fim = inicio + TAMANHO_PAGINA
    # ...


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Tipos de Dados                                                                    

# ---------------------------------------------------------------------------------------------------------------------------------------


# Tipos de dados: existem vários tipos de dados em Python, incluindo números inteiros, números de ponto flutuante, strings, 
# listas, tuplas, dicionários e conjuntos.

# Exemplo:


x = 5 # inteiro
y = 3.14 # ponto flutuante
z = "Olá" # string
my_list = [1, 2, 3] # lista
my_tuple = (4, 5, 6) # tupla
my_dict = {"nome": "Maria", "idade": 30} # dicionário
my_set = {1, 2, 3, 4} # conjunto


#                                                                ----------------------


# Dicionários (dict): são estruturas de dados que permitem armazenar valores associados a uma chave. 
# É possível acessar e modificar esses valores a partir da chave correspondente. Por exemplo:


dados_aluno = {'nome': 'João', 'idade': 20, 'curso': 'Engenharia'}
print(dados_aluno['idade']) # saída: 20


#                                                                ----------------------


# Listas (list): são sequências ordenadas de elementos que podem ser modificados. 
# É possível adicionar, remover e alterar elementos da lista. Por exemplo:


lista_frutas = ['maçã', 'banana', 'laranja']
lista_frutas.append('abacaxi') # adiciona item
print(lista_frutas) # saída: ['maçã', 'banana', 'laranja', 'abacaxi']


#                                                                ----------------------


# Conjuntos (set): são coleções de elementos únicos, ou seja, não podem ter elementos repetidos. 
# É possível realizar operações de união, interseção e diferença entre conjuntos. Por exemplo:

conjunto1 = set([1, 2, 3])
conjunto2 = set([2, 3, 4])
print(conjunto1.union(conjunto2)) # saída: {1, 2, 3, 4}


#                                                                ----------------------


# Tuplas (tuple): são sequências ordenadas de elementos imutáveis, ou seja, uma vez criada, não é possível alterar a tupla. 
# É possível acessar os elementos da tupla pelo índice. Por exemplo:

tupla_numeros = (1, 2, 3)
print(tupla_numeros[1]) # saída: 2
# Lembrando que a contaguem começa do numero 0

# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Operadores                                                                   

# ---------------------------------------------------------------------------------------------------------------------------------------


# Operadores: operadores são usados para realizar operações em valores. Os operadores em Python incluem operadores aritméticos, 
# operadores de comparação e operadores lógicos.

# Exemplo:

x = 5
y = 3


#                                                                ----------------------


# operadores aritméticos

soma = x + y    # Resultado = 8
subtracao = x - y   # Resultado = 2
multiplicacao = x * y  # Resultado = 15
divisao = x / y  # Resultado = 1.6666666666666667
resto = x % y  # Resultado = 2


#                                                                ----------------------


# operadores de comparação

maior_que = x > y
menor_que = x < y
igual_a = x == y
diferente_de = x != y


#                                                                ----------------------


# operadores lógicos

and_op = (x > 1) and (y > 1)
or_op = (x > 1) or (y < 1)
not_op = not(x > 1)


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Estruturas de controle de fluxo                                                                    

# ---------------------------------------------------------------------------------------------------------------------------------------


#Estruturas de controle de fluxo: As estruturas de controle de fluxo são usadas para controlar o fluxo de execução do programa. 
# As principais estruturas de controle de fluxo em Python são condicionais (if-else) e loops (for, while).

# Exemplo:

# estrutura condicional
x = 5

if x > 0:
    print("x é positivo")
elif x == 0:
    print("x é igual a zero")
else:
    print("x é negativo")


#                                                                ----------------------


# loop for
my_list = [1, 2, 3, 4, 5]

for item in my_list:
    print(item)


#                                                                ----------------------


# loop while
i = 0

while i < 5:
    print(i)
    i += 1


# Comando if: provavelmente o comando de controle de fluxo mais conhecido, 
# permite a execução de um bloco de código caso uma determinada condição seja atendida. Exemplo:

if idade >= 18:
  print("Você pode dirigir!")


#                                                                ----------------------


# Comando while: permite a execução repetida de um bloco de código enquanto 
# uma determinada condição for verdadeira. Exemplo:

while numero < 10:
  print(numero)
  numero += 1


#                                                                ----------------------


# Comandos for/in: permite a iteração sobre uma sequência de valores 
# (como listas, tuplas ou strings) e a execução de um bloco de código para cada valor da sequência. Exemplo:

for letra in "Python":
  print(letra)


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Funçoes                                                                     

# ---------------------------------------------------------------------------------------------------------------------------------------


#Funções: as funções são blocos de código reutilizáveis que executam uma tarefa específica. 
# Você pode definir suas próprias funções em Python usando a palavra-chave "def".

# Exemplo:

# definição de uma função
def calcula_media(notas):
    soma = sum(notas)
    media = soma / len(notas)
    return media

# chamando a função
minhas_notas = [8, 7, 9, 10]
minha_media = calcula_media(minhas_notas)
print(minha_media)


#                                                                ----------------------


# A função eval() é usada para executar operações matemáticas em números inteiros ou de ponto flutuante, mesmo em sua forma de string. 
# Isso pode ser útil se a matemática estiver no formato de string. Exemplo:

g = "(4 * 5)/4" 
d = eval(g) 
print(d) # Output: 5.0


#                                                                ----------------------


# É possível criar funções personalizadas em Python para realizar tarefas específicas. 
# Por exemplo, para calcular a média de uma lista de números, 
# podemos criar uma função com um número variável de argumentos. Veja um exemplo:

def calcular_media(*numeros):
    media = sum(numeros)/len(numeros)
    return media

lista_numeros = [1, 2, 3, 4, 5]
media = calcular_media(*lista_numeros)
print(media) # Output: 3.0


#                                                                ----------------------


# A sintaxe básica de uma função em Python consiste em nome, parâmetros e corpo. 
# O corpo da função é o bloco de código que será executado quando a função for chamada. 
# Veja um exemplo de função simples que imprime uma saudação:

def saudacao(nome):
    print(f"Olá, {nome}!")

saudacao("Maria") # Output: Olá, Maria!


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Classes & Objetos                                                                    

# ---------------------------------------------------------------------------------------------------------------------------------------


# Classes e objetos: em Python, você pode definir suas próprias classes e criar objetos com base nessas classes. 
# Uma classe é um modelo para um objeto e define seus atributos e métodos.

# Exemplo:

# definição de uma classe

class Pessoa:
    def __init__(self, nome, idade):
        self.nome = nome
        self.idade = idade

    def apresenta(self):
        print("Olá, meu nome é", self.nome, "e eu tenho", self.idade, "anos.")
        
p = Pessoa("João", 30)
p.apresenta()

# saida esperada: Olá, meu nome é João e eu tenho 30 anos.

# Este exemplo define uma classe 'Pessoanome e 'idadeidade, e um método 'apresentarapresentar que exibe uma mensagem na tela.
# utiliza o "p" para definir e printar na tela


#                                                                ----------------------


# criando um objeto

pessoa1 = Pessoa("João", 25)
pessoa1.apresentar()

#  Este exemplo cria uma instância da classe 'Pessoapessoa1 com nome "João" e idade 25, e 
# chama o método 'apresentarapresentar para exibir uma mensagem na tela.


#                                                                ----------------------


# Exemplo de classes e objetos relacionados:

# Definir a classe Retângulo com a largura e comprimento dados
class Retangulo:
    def __init__(self, comprimento, largura):
        self.comprimento = comprimento
        self.largura = largura

    # Calcular a área do retângulo
    def calcular_area(self):
        return self.comprimento * self.largura

# Definir a classe Quadrado que herda da classe Retângulo
class Quadrado(Retangulo):
    def __init__(self, lado):
        # Chamar o construtor da classe pai com o mesmo valor para largura e comprimento
        super().__init__(lado, lado)

# Criar instância de Retângulo com comprimento 4 e largura 6
retangulo1 = Retangulo(4, 6)
# Calcular a área do retângulo
area_retangulo1 = retangulo1.calcular_area()
# Imprimir a área do retângulo
print(f"Área do retângulo: {area_retangulo1}")

# Criar instância de Quadrado com lado 5
quadrado1 = Quadrado(5)
# Calcular a área do quadrado
area_quadrado1 = quadrado1.calcular_area()
# Imprimir a área do quadrado
print(f"Área do quadrado: {area_quadrado1}")


# saida esperada: Área do retângulo: 24 ; Área do quadrado: 25

#Este exemplo define duas classes 'RetanguloQuadrado, onde 'QuadradoQuadrado é uma subclasse de 'RetanguloRetangulo.
#  «RetanguloRetangulo tem um método 'calcular_areacalcular_area que calcula a área do retângulo, 
# enquanto 'QuadradoQuadrado herda o método e o construtor de 'RetanguloRetangulo para calcular a área do quadrado. 
# O código cria duas instâncias, uma de cada classe, e calcula a área de cada uma.


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Modulos & Pacotes                                                                   

# ---------------------------------------------------------------------------------------------------------------------------------------


#Módulos e pacotes: em Python, você pode usar módulos e pacotes para organizar seu código e reutilizar o código de outras pessoas. 
# Um módulo é um arquivo contendo definições e declarações em Python, enquanto um pacote é uma coleção de módulos.

# Exemplo:

# importando um módulo
import math

# usando funções do módulo math
x = math.sqrt(25)
y = math.pow(2, 3)
print(x)
print(y)

# importando um pacote
from sklearn.linear_model import LinearRegression

# criando um objeto de regressão linear
reg = LinearRegression()


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Expressões lambda                                                                   

# ---------------------------------------------------------------------------------------------------------------------------------------


# Expressões lambda: as expressões lambda são funções anônimas que podem ser definidas em uma única linha. 
# Elas são úteis quando você precisa de uma função rápida e simples.

# Exemplo:

# definição de uma expressão lambda
dobro = lambda x: x * 2

# chamando a expressão lambda
resultado = dobro(5)
print(resultado)


#                                                              ----------------------


# Ordenação de lista de tuplas: 
# Suponha que você tenha uma lista de tuplas e queira classificar a lista com base no segundo elemento de cada tupla.
#  Você pode usar a expressão lambda da seguinte forma:

lista_tuplas = [(1, 2), (3, 1), (5, 4), (2, 2)]
sorted(lista_tuplas, key=lambda x: x[1])

# O resultado será a lista ordenada [(3, 1), (1, 2), (2, 2), (5, 4)].


#                                                                ----------------------


#Filtragem de dados em uma lista: 
# Suponha que você tenha uma lista de números e queira filtrar apenas os números pares. 
# Você pode usar a expressão lambda da seguinte forma:

lista_numeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
list(filter(lambda x: x % 2 == 0, lista_numeros))

# O resultado será a lista [2, 4, 6, 8, 10].


#                                                                ----------------------


# Mapeamento de dados em uma lista: 
# Suponha que você tenha uma lista de números e queira criar uma nova lista com o quadrado de cada número. 
# Você pode usar a expressão lambda da seguinte forma:

lista_numeros = [1, 2, 3, 4, 5]
list(map(lambda x: x ** 2, lista_numeros))

# O resultado será a lista [1, 4, 9, 16, 25].


#                                                                ----------------------


# Criação de funções personalizadas: 
# Você também pode usar expressões lambda para criar funções personalizadas. 
# Por exemplo, se você quiser criar uma função que multiplique um número por um fator específico, 
# pode fazer o seguinte:

cria_funcao = lambda x, fator: x * fator
nova_funcao = cria_funcao(2, 5)

# O resultado será 10, que é o resultado da multiplicação de 2 por 5.


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Decoradores                                                                   

# ---------------------------------------------------------------------------------------------------------------------------------------


# Decoradores: um decorador é uma função que modifica a funcionalidade de outra função. 
# Eles são usados ​​para modificar ou estender o comportamento de funções existentes sem alterá-las.

# Exemplo:

# definição de um decorador: 

def meu_decorador(func):
    def wrapper():
        print("Antes da função")
        func()
        print("Depois da função")
    return wrapper

# aplicando o decorador a uma função:

@meu_decorador
def minha_funcao():
    print("Minha função")

# chamando a função com o decorador aplicado:

minha_funcao()


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Geradores                                                                  

# ---------------------------------------------------------------------------------------------------------------------------------------

#Geradores: um gerador é uma função que produz uma sequência de valores usando a palavra-chave "yield". 
# Eles são úteis para produzir grandes quantidades de dados de maneira eficiente, pois são avaliados sob demanda.

# Exemplo:

# definição de um gerador
def numeros_pares(n):
    for i in range(n):
        if i % 2 == 0:
            yield i

# usando o gerador para imprimir os primeiros 10 números pares
for num in numeros_pares(10):
    print(num)


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Tratamento de excessoes                                                                   

# ---------------------------------------------------------------------------------------------------------------------------------------


#Tratamento de exceções: o tratamento de exceções é usado para lidar com erros que ocorrem durante a execução do programa. 
# Em Python, você pode usar a palavra-chave "try" e "except" para capturar e lidar com exceções.

# Exemplo:

# tratamento de exceções
try:
    x = 10 / 0
except ZeroDivisionError:
    print("Não é possível dividir por zero")


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Automação de tarefas                                                                    

# ---------------------------------------------------------------------------------------------------------------------------------------


# Automação de tarefas: você pode usar Python para automatizar tarefas repetitivas no seu trabalho, 
# como coletar dados de várias fontes, processar arquivos em lote ou enviar e-mails em massa.

# Exemplo:

# automatizando o envio de e-mails em massa

# importando a biblioteca smtplib para enviar e-mails
import smtplib

# importando a classe MIMEText para formatar o conteúdo do e-mail
from email.mime.text import MIMEText

# criando uma mensagem de e-mail
msg = MIMEText("Olá, essa é uma mensagem de teste!")

# adicionando um assunto na mensagem
msg['Subject'] = "Testando envio de e-mails em massa"

# definindo o remetente do e-mail
msg['From'] = "seuemail@empresa.com"

# lendo a lista de destinatários do arquivo "lista_de_destinatarios.txt"
with open("lista_de_destinatarios.txt") as f:
    destinatarios = f.readlines()

# configurando as credenciais do servidor SMTP do Gmail
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login("seuemail@empresa.com", "suasenha")

# iterando sobre a lista de destinatários e enviando a mensagem para cada um
for destinatario in destinatarios:
    server.sendmail("seuemail@empresa.com", destinatario, msg.as_string())

# encerrando a conexão com o servidor SMTP
server.quit()



# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Automação de tarefas repetitivas                                                                 

# ---------------------------------------------------------------------------------------------------------------------------------------


# Automação de tarefas repetitivas:
# Baixar um arquivo CSV da web usando a biblioteca requests:

import requests

url = 'https://exemplo.com/dados.csv'
response = requests.get(url)

with open('dados.csv', 'wb') as f:
    f.write(response.content)

#                                                                ----------------------

#Automação de tarefas repetitivas:
# Baixar arquivos em lote da internet:

import requests   # importa o módulo requests para fazer solicitações HTTP
import smtplib    # importa o módulo smtplib para enviar emails

# URLs dos arquivos a serem baixados
urls = ['http://example.com/file1.txt', 'http://example.com/file2.txt', 'http://example.com/file3.txt']

# Loop através de todas as URLs
for url in urls:
    response = requests.get(url)   # envia uma solicitação HTTP GET para a URL
    with open(url.split('/')[-1], 'wb') as f:  # abre o arquivo local com o mesmo nome da URL final e modo de escrita binária
        f.write(response.content)   # escreve o conteúdo da resposta HTTP no arquivo local

# Configuração do servidor SMTP para enviar emails
server = smtplib.SMTP('smtp.gmail.com', 587)   # configura o servidor SMTP e a porta
server.starttls()   # inicia a conexão segura TLS
server.login('seuemail@gmail.com', 'suasenha')   # faz login no servidor SMTP

mensagem = "Corpo da mensagem"   # define a mensagem a ser enviada por email
server.sendmail("seuemail@gmail.com", "emaildestino@exemplo.com", mensagem)   # envia a mensagem por email
server.quit()   # finaliza a conexão com o servidor SMTP


#                                                                ----------------------

# Automação de tarefas repetitivas: Para automatizar tarefas repetitivas, você pode usar bibliotecas como Pandas e Selenium.
# Por exemplo, suponha que você precise baixar diariamente um arquivo CSV de uma API da web e, em seguida, 
# processá-lo e enviá-lo por e-mail para sua equipe. Você pode usar Python para automatizar todo esse processo. Veja um exemplo:

import pandas as pd
import requests
import smtplib

# baixar o arquivo CSV da API da web
url = 'https://exemplo.com/api/dados'
r = requests.get(url)
dados = r.content.decode('utf-8')

# transformar o CSV em um dataframe do Pandas
df = pd.read_csv(dados)

# processar o dataframe
df_processado = ...

# enviar o arquivo processado por e-mail
de = 'seuemail@gmail.com'
para = ['email1@gmail.com', 'email2@gmail.com']
msg = 'Subject: Arquivo Processado\n\n{}'.format(df_processado.to_string())

# conectar ao servidor de e-mail
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()

# fazer login no e-mail
server.login(de, 'senha_do_email')

# enviar a mensagem
server.sendmail(de, para, msg)

# desconectar do servidor de e-mail
server.quit()


# Nesse exemplo, utilizamos a biblioteca Pandas para transformar o arquivo CSV em um dataframe, processamos o dataframe e,
#  em seguida, enviamos o arquivo processado por e-mail para uma lista de destinatários usando a biblioteca smtplib.


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Análise de dados                                                                 

# ---------------------------------------------------------------------------------------------------------------------------------------


# Análise de dados:

# Importar a biblioteca pandas com o alias 'pd'
import pandas as pd

# Ler o arquivo CSV 'dados.csv' e armazenar em um objeto DataFrame chamado 'dados'
dados = pd.read_csv('dados.csv')

# Exibir as primeiras 5 linhas do DataFrame 'dados'
print(dados.head())

# Calcular a média, desvio padrão e mediana da coluna 'coluna' do DataFrame 'dados'
media = dados['coluna'].mean() # calcular a média
desvio_padrao = dados['coluna'].std() # calcular o desvio padrão
mediana = dados['coluna'].median() # calcular a mediana

# Exibir os valores calculados
print('Média:', media)
print('Desvio padrão:', desvio_padrao)
print('Mediana:', mediana)

#                                                                ----------------------

# Análise de dados:
# Importar e manipular dados com Pandas:

import pandas as pd

# Importar e ler dados do arquivo CSV
data = pd.read_csv('dados.csv')

# Remover linhas com valores ausentes (NaN)
data = data.dropna()

# Agrupar dados pela coluna 'nome' e calcular a média
data = data.groupby('nome').mean()

# Exportar os dados processados para um novo arquivo CSV
data.to_csv('dados_processados.csv')

# Visualizar dados com Matplotlib:

import matplotlib.pyplot as plt

# Definir os dados para plotagem do gráfico
x = [1, 2, 3, 4]
y = [10, 20, 30, 40]

# Plotar o gráfico com os dados definidos acima
plt.plot(x, y)

# Definir o nome do eixo x do gráfico
plt.xlabel('Eixo X')

# Definir o nome do eixo y do gráfico
plt.ylabel('Eixo Y')

# Definir o título do gráfico
plt.title('Título do Gráfico')

# Exibir o gráfico
plt.show()


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                           Machine Learning                                                                 

# ---------------------------------------------------------------------------------------------------------------------------------------

# Machine Learning:
# Criar modelo de regressão linear com Scikit-Learn:

from sklearn.linear_model import LinearRegression
import numpy as np

# Definindo as variáveis de entrada (X) e saída (y)
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([10, 20, 30, 40, 50])

# Criando um objeto LinearRegression
model = LinearRegression()

# Treinando o modelo com os dados de entrada e saída
model.fit(X, y)

# Definindo novos dados de entrada para previsão
x_test = np.array([[6]])

# Prevendo a saída correspondente aos novos dados de entrada
y_pred = model.predict(x_test)

# Imprimindo o resultado da previsão
print(y_pred)



# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                   Processamento de Linguagem Natural (NLP)                                                                 

# ---------------------------------------------------------------------------------------------------------------------------------------


# Processamento de Linguagem Natural (NLP): Python é uma das linguagens mais populares para o processamento de linguagem natural. 
# Você pode usar bibliotecas como NLTK, spaCy e TextBlob para realizar tarefas como tokenização, stemming, lematização, 
# ]análise de sentimento e identificação de entidades.

# Exemplo:

# importar bibliotecas
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# baixar recursos adicionais
nltk.download('punkt')
nltk.download('stopwords')

# definir uma frase de exemplo
texto = "O processamento de linguagem natural é uma área importante em ciência da computação."

# tokenização
tokens = word_tokenize(texto)

# remoção de stop words
stop_words = set(stopwords.words('portuguese'))
tokens_sem_stop = [w for w in tokens if not w in stop_words]

# stemming
stemmer = PorterStemmer()
tokens_stem = [stemmer.stem(w) for w in tokens_sem_stop]

# impressão dos resultados
print(tokens)
print(tokens_sem_stop)
print(tokens_stem)

# Neste exemplo, nós utilizamos as bibliotecas NLTK e nltk.corpus para baixar recursos adicionais, tokenizar 
# e remover stop words de uma frase de exemplo em português. 
# Em seguida, utilizamos a biblioteca nltk.stem para aplicar a técnica de stemming nos tokens resultantes.

# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                   Criação de relatórios                                                                 

# ---------------------------------------------------------------------------------------------------------------------------------------


#Criação de relatórios: Para criar relatórios, você pode usar bibliotecas como Pandas, Matplotlib e ReportLab. 
# Por exemplo, suponha que você precise criar um relatório mensal de vendas em PDF, com gráficos e tabelas. 
# Você pode usar Python para criar esse relatório. Veja um exemplo:


import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# carregar os dados das vendas em um dataframe do Pandas
df_vendas = pd.read_csv('vendas.csv')

# criar um gráfico de barras das vendas por mês
df_vendas_por_mes = df_vendas.groupby('mes')['vendas'].sum()
df_vendas_por_mes.plot(kind='bar')

# salvar o gráfico como uma imagem PNG
plt.savefig('grafico_vendas.png')

# criar um relatório em PDF com o gráfico
pdf = canvas.Canvas('relatorio_vendas.pdf', pagesize=letter)
pdf.drawString(100, 750, "Relatório de Vendas")

# adicionar o gráfico ao PDF
pdf.drawInlineImage('grafico_vendas.png', 100, 500)

# adicionar um texto explicativo
pdf.drawString(100, 400, "Este gráfico mostra as vendas por mês")

# salvar e fechar o PDF
pdf.save()

# Neste exemplo, utilizamos a biblioteca ReportLab para criar um relatório em PDF com um gráfico de barras das vendas por mês. 
# Primeiro, carregamos os dados das vendas em um dataframe do Pandas e criamos o gráfico de barras utilizando o método plot() do Pandas. 
# Em seguida, salvamos o gráfico como uma imagem PNG utilizando o método savefig() do Matplotlib. 
# Depois, criamos um novo arquivo PDF utilizando a biblioteca ReportLab e adicionamos o gráfico ao PDF utilizando o método drawInlineImage(). 
# Por fim, adicionamos um texto explicativo e salvamos o arquivo PDF utilizando o método save().


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                   Buscas no google e criançao de relatorio                                                                

# ---------------------------------------------------------------------------------------------------------------------------------------


# Aqui está um exemplo de código que realiza uma busca no Google sobre um determinado assunto e gera um relatório no final 
# usando a biblioteca BeautifulSoup em Python:

import requests
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def search_google(query):
    # realizar a busca no Google
    response = requests.get("https://www.google.com/search?q=" + query)
    soup = BeautifulSoup(response.content, "html.parser")

    # extrair os resultados da busca
    results = []
    for item in soup.select(".g"):
        # extrair o título do resultado da busca
        title = item.select(".r")[0].get_text()
        # extrair o link do resultado da busca
        link = item.select(".r")[0].find("a")["href"]
        # extrair a descrição do resultado da busca
        description = item.select(".s")[0].get_text()
        # adicionar os resultados à lista de resultados
        results.append((title, link, description))

    return results

def generate_report(query, results):
    # criar um relatório em PDF com os resultados da busca
    filename = query.replace(" ", "_") + ".pdf"
    doc = canvas.Canvas(filename, pagesize=letter)

    # adicionar o título do relatório
    doc.setFont("Helvetica-Bold", 16)
    doc.drawString(72, 720, "Resultados da busca por \"" + query + "\":")

    # adicionar os resultados da busca
    doc.setFont("Helvetica", 12)
    y = 700
    for title, link, description in results:
        # adicionar o título do resultado
        doc.drawString(72, y, title)
        doc.setFont("Helvetica-Oblique", 10)
        # adicionar o link do resultado
        doc.drawString(72, y - 20, link)
        doc.setFont("Helvetica", 12)
        # adicionar a descrição do resultado
        doc.drawString(72, y - 40, description)
        y -= 80

    # salvar o arquivo PDF
    doc.save()

# exemplo de uso
query = "python tutorial"
results = search_google(query)
generate_report(query, results)



# Neste exemplo, a função 'search_google recebe uma string contendo o termo a ser pesquisado no Google e 
# retorna uma lista de tuplas contendo o título, o link e a descrição de cada resultado da pesquisa. 
# Para isso, usamos a biblioteca requests para enviar uma requisição GET para a URL da pesquisa e a biblioteca BeautifulSoup 
# para analisar o HTML da página de resultados e extrair as informações relevantes.

# Em seguida, a função 'generate_report

# Por fim, o exemplo de uso mostra como chamar as funções 'search_googlegenerate_report() com um termo de pesquisa de exemplo e 
# salvar o relatório em PDF com um nome de arquivo baseado no termo de pesquisa.


# ---------------------------------------------------------------------------------------------------------------------------------------

#                                                               BOA SORTE                                                                

# ---------------------------------------------------------------------------------------------------------------------------------------
 


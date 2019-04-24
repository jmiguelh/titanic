import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# lendo arquivos
treino = pd.read_csv('train.csv')
teste = pd.read_csv('test.csv')

# Limpando dataset
treino.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
teste.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Tratando dados 
treino2 = pd.get_dummies(treino)
teste2 = pd.get_dummies(teste)
#treino2.isnull().sum().sort_values(ascending=False).head(10)
treino2['Age'].fillna(treino2['Age'].mean(),inplace=True)
teste2['Age'].fillna(teste2['Age'].mean(),inplace=True)
treino2['Fare'].fillna(treino2['Fare'].mean(),inplace=True)
teste2['Fare'].fillna(teste2['Fare'].mean(),inplace=True)


# Criação do modelo 
x = treino2.drop('Survived', axis=1)
y = treino2['Survived'] 

tree = DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(x,y)

# Validar precisão
tree.score(x,y)

# Criando resultado
resultado = pd.DataFrame()
resultado['PassengerId'] = teste2['PassengerId']
resultado['Survived'] = tree.predict(teste2)
resultado.to_csv('resultado.csv', index=False)

from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

dot_data = StringIO()
export_graphviz(tree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
display(Image(graph.create_png()))

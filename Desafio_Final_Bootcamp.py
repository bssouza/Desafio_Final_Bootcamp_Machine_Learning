# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
#%%
# Importando dependencias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy.stats as ss
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix

# %%
# Abrindo arquivo
data = pd.read_csv("cars.csv")

# %%
# Apresentando as 5 primeiras linhas do dataset
data.head()

# %%
# PERGUNTA 1 - Após a utilização da biblioteca pandas para a leitura dos dados sobre os valores lidos, é CORRETO afirmar:
data.info()

# %%
# PERGUNTA 1 - Após a utilização da biblioteca pandas para a leitura dos dados sobre os valores lidos, é CORRETO afirmar:
data.isnull().sum()

# %%
#Convertendo colunas em numeric
data["cubicinches"] = pd.to_numeric(data["cubicinches"],errors='coerce')
data["weightlbs"] = pd.to_numeric(data["weightlbs"],errors='coerce')

# %%
# PERGUNTA 3 - Indique quais eram os índices dos valores presentes no dataset que "forçaram" o pandas a compreender a variável "cubicinches" como string.
data[pd.to_numeric(data['cubicinches'], errors='coerce').isnull()]

# %%
# PERGUNTA 4 - Após a transformação das variáveis "string" para os valores numéricos, quantos valores nulos (células no dataframe) passaram a existir no dataset?
data["cubicinches"].isnull().sum() + data["weightlbs"].isnull().sum()

# %%
# PERGUNTA 5 - Substitua os valores nulos introduzidos no dataset, após a transformação, pelo valor médio das colunas. Qual é o novo valor médio da coluna "weightlbs"?
data = data.fillna(data.mean())
round(data["weightlbs"].mean(),2)

# %%
# PERGUNTA 6 - Após substituir os valores nulos pela média das colunas, selecione as colunas ['mpg', 'cylinders', 'cubicinches', 'hp', 'weightlbs', 'time-to-60', 'year']. Qual é o valor da mediana para a característica 'mpg'?
data["mpg"].median()
#data["hp"].median()

# %%
# PERGUNTA 7 - Qual é a afirmação CORRETA sobre o valor de 14,00 para a variável "time-to-60"?
data["time-to-60"].describe()


# %%
# PERGUNTA 8 - Sobre o coeficiente de correlação de Pearson entre as variáveis "cylinders" e "mpg", é correto afirmar, EXCETO:
np.corrcoef(data["cylinders"], data["mpg"])[0, 1]

# %%
# PERGUNTA 9 -Sobre o boxplot da variável "hp", é correto afirmar, EXCETO:
boxplot = data.boxplot(column=['hp'])  #constroi o boxplot para as colunas desejadas

# %%
# PERGUNTA 10 - Após normalizado, utilizando a função StandardScaler(), qual é o maior valor para a variável "hp"?
df = pd.DataFrame(data, columns=['mpg','cylinders','cubicinches','hp','weightlbs','time-to-60','year']).astype('float64')
scaled_features = StandardScaler().fit_transform(df)
scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)
col_stats = ss.describe(scaled_features_df['hp'])
(_min,_max) = col_stats[1] # [Min/Max][Max Value]
_max

# %%
# PERGUNTA 11 - Aplicando o PCA, conforme a definição acima, qual é o valor da variância explicada com pela primeira componente principal?
pca = decomposition.PCA(n_components=7, svd_solver='full')
pca.fit_transform(scaled_features_df)
round(pca.explained_variance_ratio_.cumsum()[0]*100)

# %%
# PERGUNTA 12 - Utilize os três primeiros componentes principais para construir o K-means com um número de 3 clusters. Sobre os clusters, é INCORRETO afirmar:
#KMeans(n_clusters=3,random_state=42)
df = pd.DataFrame(data, columns=['mpg','cylinders','cubicinches']).astype('float64')
kmeans = KMeans(n_clusters=3, random_state=42).fit_predict(df)
pd.Series(kmeans).value_counts()

# %%
# Pergunta 13 - Após todo o processamento realizado nos itens anteriores, crie uma coluna que contenha a variável de eficiência do veículo. 
# Veículos que percorrem mais de 25 milhas com um galão (“mpg”>25) devem ser considerados eficientes. 
# Utilize as colunas  ['cylinders' ,'cubicinches' ,'hp' ,'weightlbs','time-to-60'] como entradas e como saída a coluna de eficiência criada. 
# Utilizando a árvore de decisão como mostrado, qual é a acurácia do modelo?
data['eficiente'] = np.where(data['mpg']>25,1,0)
df = pd.DataFrame(data, columns=['cylinders','cubicinches','hp','weightlbs','time-to-60']).astype('float64')
X = StandardScaler().fit_transform(df)
Y = data['eficiente']

X_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)
model_reg = LogisticRegression(random_state=42).fit(X_train,y_train)
y_pred = model_reg.predict(x_test)
classification_report(y_test, y_pred)

# %%
estimator = DecisionTreeClassifier(random_state=42)
estimator.fit(X_train, y_train)
y_pred_2 = estimator.predict(x_test)
classification_report(y_test, y_pred_2)

# %%
# PERGUNTA 14 - Sobre a matriz de confusão obtida após a aplicação da árvore de decisão, 
# como mostrado anteriormente, é INCORRETO afirmar:
sns.heatmap(confusion_matrix(y_test ,y_pred_2), annot = True)

# %%
# PERGUNTA 15 - Utilizando a mesma divisão de dados entre treinamento e teste empregada para a análise anterior, 
# aplique o modelo de regressão logística como mostrado na descrição do trabalho. 
# Comparando os resultados obtidos com o modelo de árvore de decisão, é INCORRETO afirmar:

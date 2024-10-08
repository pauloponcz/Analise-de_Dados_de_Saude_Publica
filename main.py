import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

# Carregar o dataset a partir do arquivo local
data = pd.read_csv('healthcare-dataset-stroke-data.csv')  # Certifique-se de que o arquivo está na mesma pasta que o script

# 1. Limpeza de Dados
# Verificar valores faltantes
print("Valores nulos por coluna:")
print(data.isnull().sum())

# A coluna 'bmi' tem valores 'N/A' que precisam ser tratados
# Substituir 'N/A' por NaN para facilitar o tratamento
data['bmi'].replace('N/A', np.nan, inplace=True)

# Preencher os valores nulos da coluna 'bmi' com a mediana
imputer = SimpleImputer(strategy='median')
data['bmi'] = imputer.fit_transform(data[['bmi']])

# Mapear as colunas categóricas de hipertensão e doenças cardíacas para rótulos descritivos
data['hypertension'] = data['hypertension'].map({0: 'Sem Hipertensão', 1: 'Com Hipertensão'})
data['heart_disease'] = data['heart_disease'].map({0: 'Sem Doença Cardíaca', 1: 'Com Doença Cardíaca'})
data['stroke'] = data['stroke'].map({0: 'Sem AVC', 1: 'Com AVC'})

# Aplicando One-Hot Encoding nas colunas categóricas
data_encoded = pd.get_dummies(data, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease', 'stroke'])

# Verificar as colunas geradas após o One-Hot Encoding
print("Colunas geradas pelo One-Hot Encoding:")
print(data_encoded.columns)

# 2. Análise de Fatores de Risco com gráficos
# Ajustar os gráficos com os nomes corretos das colunas após One-Hot Encoding
plt.figure(figsize=(12, 6))
sns.countplot(x=data['hypertension'], hue=data['stroke'])
plt.title('Relação entre Hipertensão e AVC')
plt.xlabel('Hipertensão')
plt.ylabel('Contagem')
plt.show()

plt.figure(figsize=(12, 6))
sns.countplot(x=data['heart_disease'], hue=data['stroke'])
plt.title('Relação entre Doenças Cardíacas e AVC')
plt.xlabel('Doença Cardíaca')
plt.ylabel('Contagem')
plt.show()

# 3. Histograma de Distribuição de Idades
plt.figure(figsize=(12, 6))
sns.histplot(data[data['stroke'] == 'Com AVC']['age'], bins=30, kde=True)
plt.title('Distribuição de Idade dos Pacientes que Sofreram AVC')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

# 4. Modelagem Simples
# Separar variáveis independentes (X) e dependente (y)
X = data_encoded.drop(columns=['id'])  # Excluindo 'id' do X (features)
y = data_encoded['stroke_Com AVC']  # A variável alvo é 'stroke_Com AVC'

# Dividir o dataset em treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizar as colunas numéricas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Modelo de Regressão Logística
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)

# Previsões no conjunto de teste
y_pred = log_reg.predict(X_test)

# Avaliação do modelo - Regressão Logística
print("Acurácia do modelo de Regressão Logística:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação - Regressão Logística:\n", classification_report(y_test, y_pred))

# Matriz de confusão - Regressão Logística
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - Regressão Logística')
plt.show()

# Modelo Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Avaliação do modelo - Random Forest
print("Acurácia do modelo Random Forest:", accuracy_score(y_test, y_pred_rf))
print("Relatório de Classificação - Random Forest:\n", classification_report(y_test, y_pred_rf))

# Matriz de Confusão - Random Forest
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(conf_matrix_rf, annot=True, fmt='d', cmap='Greens')
plt.title('Matriz de Confusão - Random Forest')
plt.show()

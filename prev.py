# Importar bibliotecas necessárias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

print("Iniciando o projeto de Previsão de Preços de Imóveis em São Paulo...")

# --- 1. Carregamento e Simulação dos Dados ---
print("\n1. Criando um conjunto de dados simulado para São Paulo...")

np.random.seed(42) # Para reprodutibilidade

# Número de amostras
n_samples = 1000

# Gerar dados para as características
data = {
    'Area_m2': np.random.randint(40, 300, n_samples),
    'Num_Quartos': np.random.randint(1, 5, n_samples),
    'Num_Banheiros': np.random.randint(1, 4, n_samples),
    'Num_Vagas': np.random.randint(0, 3, n_samples),
    'Idade_Imovel_Anos': np.random.randint(0, 50, n_samples),
    'Bairro': np.random.choice(['Pinheiros', 'Jardins', 'Moema', 'Vila Madalena', 'Santana', 'Tatuapé', 'Morumbi', 'Capão Redondo'], n_samples),
    'Tipo_Imovel': np.random.choice(['Apartamento', 'Casa', 'Terreno'], n_samples, p=[0.7, 0.25, 0.05]),
    'Proximidade_Metro_km': np.random.rand(n_samples) * 5 # De 0 a 5 km
}

df = pd.DataFrame(data)

# Gerar preço (variável alvo) com base nas características, adicionando algum ruído
# Uma fórmula simples para simular preços:
# Preço = f(Area, Quartos, Banheiros, Localização) + Ruído
preco_base = (df['Area_m2'] * 5000 +
              df['Num_Quartos'] * 50000 +
              df['Num_Banheiros'] * 30000 +
              df['Num_Vagas'] * 20000)

# Ajuste de preço por bairro (exemplo: Pinheiros e Jardins mais caros)
bairro_multiplicador = {
    'Pinheiros': 1.8, 'Jardins': 2.0, 'Moema': 1.7, 'Vila Madalena': 1.6,
    'Santana': 1.2, 'Tatuapé': 1.3, 'Morumbi': 1.5, 'Capão Redondo': 0.8
}
df['Multiplicador_Bairro'] = df['Bairro'].map(bairro_multiplicador)
preco_base = preco_base * df['Multiplicador_Bairro']

# Ajuste por tipo de imóvel
tipo_multiplicador = {'Apartamento': 1.0, 'Casa': 1.2, 'Terreno': 0.6}
df['Multiplicador_Tipo'] = df['Tipo_Imovel'].map(tipo_multiplicador)
preco_base = preco_base * df['Multiplicador_Tipo']

# Ajuste por proximidade do metrô
preco_base = preco_base - (df['Proximidade_Metro_km'] * 10000) # Quanto mais perto, maior o preço

# Adicionar aleatoriedade (ruído) e garantir preços positivos
df['Preco_RS'] = np.maximum(50000, preco_base + np.random.normal(0, 150000, n_samples))
df.drop(['Multiplicador_Bairro', 'Multiplicador_Tipo'], axis=1, inplace=True) # Remover colunas auxiliares

print("Conjunto de dados de São Paulo simulado com sucesso.")
print("Primeiras 5 linhas do dataset:")
print(df.head())
print("\nInformações do dataset:")
df.info()
print("\nEstatísticas descritivas:")
print(df.describe())

# --- 2. Pré-processamento de Dados ---
print("\n2. Pré-processando os dados...")

# Verificar valores ausentes (no caso de dados reais)
print("Verificando valores ausentes:")
print(df.isnull().sum())

# Separar variáveis independentes (X) e dependente (y)
X = df.drop('Preco_RS', axis=1)
y = df['Preco_RS']

# Definir colunas numéricas e categóricas
numeric_features = ['Area_m2', 'Num_Quartos', 'Num_Banheiros', 'Num_Vagas', 'Idade_Imovel_Anos', 'Proximidade_Metro_km']
categorical_features = ['Bairro', 'Tipo_Imovel']

# Criar um pipeline de pré-processamento para diferentes tipos de colunas
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features), # Escalonar colunas numéricas
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # One-Hot Encoding para categóricas
    ])

# Dividir os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Dados de treino: {X_train.shape[0]} amostras")
print(f"Dados de teste: {X_test.shape[0]} amostras")

# --- 3. Treinamento do Modelo ---
print("\n3. Treinando o modelo de Regressão Linear...")

# Criar um pipeline completo que inclui pré-processamento e o modelo
model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('regressor', LinearRegression())])

# Treinar o modelo com os dados de treino
model_pipeline.fit(X_train, y_train)

print("Modelo de Regressão Linear treinado.")

# --- 4. Avaliação do Modelo ---
print("\n4. Avaliando o desempenho do modelo...")

# Fazer previsões no conjunto de teste
y_pred = model_pipeline.predict(X_test)

# Calcular as métricas de avaliação
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Visualização das previsões vs. valores reais
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # Linha ideal (previsão = real)
plt.xlabel("Preços Reais (R$)")
plt.ylabel("Preços Previstos (R$)")
plt.title("Preços Reais vs. Preços Previstos em São Paulo")
plt.grid(True)
plt.show()

# Resíduos (diferença entre previsto e real)
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=30, color='skyblue')
plt.title("Distribuição dos Resíduos")
plt.xlabel("Resíduos (Real - Previsto)")
plt.ylabel("Frequência")
plt.grid(True)
plt.show()

print("\nProjeto concluído! Explore os gráficos para entender melhor o desempenho do modelo.")

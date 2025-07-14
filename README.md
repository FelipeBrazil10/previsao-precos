# Projeto: Previsão de Preços de Imóveis em São Paulo com Machine Learning

Este é um projeto de Machine Learning focado em construir um modelo de regressão para prever o preço de imóveis na cidade de **São Paulo, Brasil**. Ele demonstra um pipeline completo de ML, desde a simulação e pré-processamento dos dados até o treinamento e a avaliação do modelo.

## 🚀 Tecnologias Utilizadas

* **Python**
* **pandas**: Para manipulação e análise de dados.
* **numpy**: Para operações numéricas.
* **scikit-learn**: Para Machine Learning (pré-processamento, modelo, métricas).
* **matplotlib**: Para visualização de dados.
* **seaborn**: Para visualizações estatísticas atraentes.

## 📊 Conjunto de Dados

Para este projeto, um **conjunto de dados de imóveis em São Paulo foi simulado**. Ele contém diversas características comuns do mercado imobiliário paulista, como área, número de quartos, vagas de garagem, idade do imóvel, tipo (apartamento, casa, terreno) e, crucialmente, o bairro e a proximidade com o metrô. A variável alvo é o preço do imóvel em Reais (R$).

**Observação:** Em um projeto real, os dados seriam coletados de fontes como portais imobiliários (ZAP Imóveis, Viva Real, etc.) ou outras bases de dados específicas do mercado. A simulação aqui serve para demonstrar a estrutura e o fluxo do projeto.

## ✨ Características Principais

* **Simulação de Dados:** Geração programática de um dataset que reflete características do mercado de São Paulo.
* **Pré-processamento de Dados:** Tratamento de dados, separação em conjuntos de treino/teste, escalonamento de features numéricas e codificação One-Hot para variáveis categóricas (bairro, tipo de imóvel).
* **Treinamento de Modelo:** Utilização de um modelo de **Regressão Linear**.
* **Avaliação de Desempenho:** Cálculo de métricas como **Mean Squared Error (MSE)**, **Root Mean Squared Error (RMSE)** e **R-squared (R²)** para avaliar a performance do modelo.
* **Visualização de Resultados:** Geração de gráficos para comparar preços reais com os previstos e analisar a distribuição dos resíduos.

## ⚙️ Como Executar o Projeto

Siga os passos abaixo para rodar este projeto em sua máquina local:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/seu-usuario/seu-projeto-github.git](https://github.com/seu-usuario/seu-projeto-github.git)
    cd seu-projeto-github
    ```
2.  **Crie e ative um ambiente virtual (recomendado):**
    ```bash
    python -m venv venv
    # No Windows:
    .\venv\Scripts\activate
    # No macOS/Linux:
    source venv/bin/activate
    ```
3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Execute o script principal:**
    ```bash
    python main.py
    ```

Ao executar o script, você verá as etapas do processo no console e serão gerados dois gráficos: um de dispersão comparando preços reais com previstos e um histograma da distribuição dos resíduos.

## 📈 Resultados (Exemplo)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Função para converter strings com formato brasileiro para números corretos
def converter_para_float(valor):
    return float(valor.replace('.', '').replace(',', '.'))


# Carregar o arquivo CSV com as devidas correções de formato
df = pd.read_csv('data2.csv', converters={'NFocos': converter_para_float, 'AreaD': converter_para_float})


# Função para análise de tendência com regressão linear
def analisar_tendencia(df, coluna, nome_coluna):
    X = df[['ANO']].values
    y = df[[coluna]].values

    # Modelo de regressão linear
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Fazer previsões
    y_pred = modelo.predict(X)

    # Plotar gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(df['ANO'], y, label=f'Valores reais {nome_coluna}', marker='o')
    plt.plot(df['ANO'], y_pred, label=f'Tendência {nome_coluna}', linestyle='--')
    plt.xlabel('Ano')
    plt.ylabel(nome_coluna)
    plt.title(f'Análise de Tendência: {nome_coluna} ao longo dos anos')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Exibir coeficiente angular (inclinação) para ver a tendência
    print(f"Tendência de {nome_coluna}: Inclinação = {modelo.coef_[0][0]}")


# Analisar tendência para NFocos
analisar_tendencia(df, 'AreaD', 'Número de Focos de Incêndio')

# Analisar tendência para AreaD
analisar_tendencia(df, 'FocosN', 'Área Desmatada')

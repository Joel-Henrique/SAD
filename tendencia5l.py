import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Função para converter strings com formato brasileiro para números corretos
def converter_para_float(valor):
    return float(valor.replace('.', '').replace(',', '.'))


# Carregar o arquivo CSV com as devidas correções de formato
df = pd.read_csv('data3.csv', converters={'NFocos': converter_para_float, 'AreaD': converter_para_float})


# Função para calcular soma e média em janelas deslizantes de 5 anos
def calcular_soma_media_janela(df, janela=5):
    # Criar uma lista para armazenar os resultados
    resultados = []

    # Aplicar janela deslizante de 5 anos
    for i in range(len(df) - janela + 1):
        # Selecionar a janela de 5 anos
        janela_df = df.iloc[i:i + janela]

        # Calcular a soma e a média
        soma_nfocos = janela_df['NFocos'].sum()
        soma_aread = janela_df['AreaD'].sum()
        media_nfocos = janela_df['NFocos'].mean()
        media_aread = janela_df['AreaD'].mean()

        # Guardar os resultados
        ano_inicial = janela_df['ANO'].iloc[0]
        ano_final = janela_df['ANO'].iloc[-1]
        resultados.append({
            'Intervalo': f'{ano_inicial}-{ano_final}',
            'Média NFocos': media_nfocos,
            'Média AreaD': media_aread
        })

    # Criar um DataFrame a partir dos resultados
    df_resultados = pd.DataFrame(resultados)
    return df_resultados


# Calcular soma e média para NFocos e AreaD em janelas de 5 anos
resultados = calcular_soma_media_janela(df)

# Exibir os resultados
print(resultados)


# Função para calcular a regressão linear para médias por intervalo
def calcular_tendencia_geral(resultados):
    # Criar índice baseado nos intervalos de 5 anos (para simular o tempo)
    X = np.arange(len(resultados)).reshape(-1, 1)

    # Regressão para Média NFocos
    y_nfocos = resultados['Média NFocos'].values
    modelo_nfocos = LinearRegression()
    modelo_nfocos.fit(X, y_nfocos)
    tendencia_nfocos = modelo_nfocos.predict(X)

    # Regressão para Média AreaD
    y_aread = resultados['Média AreaD'].values
    modelo_aread = LinearRegression()
    modelo_aread.fit(X, y_aread)
    tendencia_aread = modelo_aread.predict(X)

    # Imprimir coeficientes da tendência de NFocos
    print(f"Tendência para NFocos:")
    print(f"  Coeficiente de inclinação (tendência): {modelo_nfocos.coef_[0]:.4f}")
    print(f"  Intercepto: {modelo_nfocos.intercept_:.4f}\n")

    # Imprimir coeficientes da tendência de AreaD
    print(f"Tendência para AreaD:")
    print(f"  Coeficiente de inclinação (tendência): {modelo_aread.coef_[0]:.4f}")
    print(f"  Intercepto: {modelo_aread.intercept_:.4f}\n")

    return tendencia_nfocos, tendencia_aread


# Plotar as tendências e médias calculadas
def plotar_tendencias(resultados, tendencia_nfocos, tendencia_aread):
    # Criar figura e subplots
    plt.figure(figsize=(12, 6))

    # Gráfico para NFocos
    plt.subplot(1, 2, 1)
    plt.plot(resultados['Intervalo'], resultados['Média NFocos'], marker='o', linestyle='--', color='blue', label='Média NFocos')
    plt.plot(resultados['Intervalo'], tendencia_nfocos, color='red', label='Tendência NFocos')
    plt.xlabel('Intervalo de 5 Anos')
    plt.ylabel('Média de Focos de Incêndio')
    plt.title('Média de Focos de Incêndio a cada 5 Anos')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Gráfico para AreaD
    plt.subplot(1, 2, 2)
    plt.plot(resultados['Intervalo'], resultados['Média AreaD'], marker='o', linestyle='--', color='green', label='Média AreaD')
    plt.plot(resultados['Intervalo'], tendencia_aread, color='red', label='Tendência AreaD')
    plt.xlabel('Intervalo de 5 Anos')
    plt.ylabel('Média da Área Queimada (em milhões)')
    plt.title('Média da Área Desmatamento a cada 5 Anos (em milhões)')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Ajustar layout
    plt.tight_layout()
    plt.show()


# Calcular tendências para NFocos e AreaD
tendencia_nfocos, tendencia_aread = calcular_tendencia_geral(resultados)

# Plotar as tendências
plotar_tendencias(resultados, tendencia_nfocos, tendencia_aread)

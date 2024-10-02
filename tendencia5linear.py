import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
def converter_para_float(valor):
    return float(valor.replace('.', '').replace(',', '.'))
df = pd.read_csv('data2.csv', converters={'NFocos': converter_para_float, 'AreaD': converter_para_float})


def calcular_soma_media_janela(df, janela=5):
    resultados = []
    for i in range(len(df) - janela + 1):
        janela_df = df.iloc[i:i + janela]
        soma_nfocos = janela_df['NFocos'].sum()
        soma_aread = janela_df['AreaD'].sum()
        media_nfocos = janela_df['NFocos'].mean()
        media_aread = janela_df['AreaD'].mean()
        ano_inicial = janela_df['ANO'].iloc[0]
        ano_final = janela_df['ANO'].iloc[-1]
        resultados.append({
            'Intervalo': f'{ano_inicial}-{ano_final}',
            'Soma NFocos': soma_nfocos,
            'Média NFocos': media_nfocos,
            'Soma AreaD': soma_aread,
            'Média AreaD': media_aread
        })
    df_resultados = pd.DataFrame(resultados)
    return df_resultados
# Calcular soma e média para NFocos e AreaD em janelas de 5 anos
resultados = calcular_soma_media_janela(df)
print(resultados)
# Função para plotar as tendências de regressão linear
def plotar_tendencias(resultados):
    plt.figure(figsize=(12, 6))
    # Gráfico para NFocos
    plt.subplot(1, 2, 1)
    plt.plot(resultados['Intervalo'], resultados['Média NFocos'], marker='o', linestyle='--', color='blue')
    plt.xlabel('Intervalo de 5 Anos')
    plt.ylabel('Média de Focos de Incêndio')
    plt.title('Média de Focos de Incêndio a cada 5 Anos')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Gráfico para AreaD
    plt.subplot(1, 2, 2)
    plt.plot(resultados['Intervalo'], resultados['Média AreaD'], marker='o', linestyle='--', color='green')
    plt.xlabel('Intervalo de 5 Anos')
    plt.ylabel('Média da Área Queimada (em milhões)')
    plt.title('Média da Área Queimada a cada 5 Anos (em milhões)')
    plt.xticks(rotation=45)
    plt.grid(True)

    # Regressão Linear para NFocos
    X_nfocos = np.arange(len(resultados)).reshape(-1, 1)
    y_nfocos = resultados['Média NFocos'].values
    modelo_nfocos = LinearRegression()
    modelo_nfocos.fit(X_nfocos, y_nfocos)
    tendencia_nfocos = modelo_nfocos.predict(X_nfocos)
    plt.plot(resultados['Intervalo'], tendencia_nfocos, color='red', label='Tendência')
    plt.legend()

    # Regressão Linear para AreaD
    X_aread = np.arange(len(resultados)).reshape(-1, 1)
    y_aread = resultados['Média AreaD'].values
    modelo_aread = LinearRegression()
    modelo_aread.fit(X_aread, y_aread)
    tendencia_aread = modelo_aread.predict(X_aread)
    plt.plot(resultados['Intervalo'], tendencia_aread, color='red', label='Tendência')
    plt.legend()
    # Ajustar layout
    plt.tight_layout()
    plt.show()


# Plotar as tendências com regressão linear
plotar_tendencias(resultados)

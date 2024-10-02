import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# Função para converter strings com formato brasileiro para números corretos
def converter_para_float(valor):
    return float(valor.replace('.', '').replace(',', '.'))


# Carregar o arquivo CSV com as devidas correções de formato
df = pd.read_csv('data2.csv', converters={'NFocos': converter_para_float, 'AreaD': converter_para_float})


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

        media_aread_milhao = media_aread * 1_000_000

        # Guardar os resultados
        ano_inicial = janela_df['ANO'].iloc[0]
        ano_final = janela_df['ANO'].iloc[-1]
        resultados.append({
            'Intervalo': f'{ano_inicial}-{ano_final}',
            'Soma NFocos': soma_nfocos,
            'Média NFocos': media_nfocos,
            'Soma AreaD': soma_aread,
            'Média AreaD': media_aread
        })

    # Criar um DataFrame a partir dos resultados
    df_resultados = pd.DataFrame(resultados)
    return df_resultados


# Calcular soma e média para NFocos e AreaD em janelas de 5 anos
resultados = calcular_soma_media_janela(df)

# Exibir os resultados
print(resultados)

# Plotar as médias de NFocos e AreaD
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(resultados['Intervalo'], resultados['Média NFocos'], marker='o', linestyle='--', color='blue')
plt.xlabel('Intervalo de 5 Anos')
plt.ylabel('Média de Focos de Incêndio')
plt.title('Média de Focos de Incêndio a cada 5 Anos')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(resultados['Intervalo'], resultados['Média AreaD'], marker='o', linestyle='--', color='green')
plt.xlabel('Intervalo de 5 Anos')
plt.ylabel('Média da Área Queimada (em milhões)')
plt.title('Média da Área Queimada a cada 5 Anos (em milhões) ')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()

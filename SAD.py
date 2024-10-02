import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv("data3.csv", thousands='.', decimal=',')
print(data)
correlation = data.corr()
plot = sns.heatmap(correlation, annot=True, fmt=".1f", linewidths=.6)
plt.show()
import pandas as pd


pd.set_option('max_columns', None)

linkpred = pd.read_csv('tables_gnn/kernel_pca/pip_linkpred_table200.csv', na_filter = False)

etapas = {}
for etapa, datos in linkpred.groupby('Etapa'):
    etapas[etapa] = datos

print('Resultados ENTRENAMIENTO de modelo de link prediction: Ordenado por precision')
print(etapas['train'].sort_values(by='Precision', ascending=False))
print('------------------------------------------------------------------------------')
print('Resultados VALIDACION de modelo de link prediction: Ordenado por precision')
print(etapas['validate'].sort_values(by='Precision', ascending=False))
print('------------------------------------------------------------------------------')
print('Resultados PRUEBA de modelo de link prediction: Ordenado por precision')
print(etapas['test'].sort_values(by='Precision', ascending=False))
print('------------------------------------------------------------------------------')

# edge_reg = pd.read_csv('tables_gnn/kernel_pca/pip_edge_regression_table50.csv', na_filter = False)

# etapas = {}
# for etapa, datos in edge_reg.groupby('Etapa'):
#     etapas[etapa] = datos

# print('Resultados ENTRENAMIENTO de modelo de edge regression: Ordenado por perdida')
# print(etapas['train'].sort_values(by='Perdida(MSE)'))
# print('------------------------------------------------------------------------------')
# print('Resultados VALIDACION de modelo de edge regression: Ordenado por perdida')
# print(etapas['validate'].sort_values(by='Perdida(MSE)'))
# print('------------------------------------------------------------------------------')
# print('Resultados PRUEBA de modelo de edge regression: Ordenado por perdida')
# print(etapas['test'].sort_values(by='Perdida(MSE)'))
# print('------------------------------------------------------------------------------')

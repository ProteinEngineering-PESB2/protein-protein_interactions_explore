import sys
import dgl
import os
import torch
import random
import time
import torch.nn as nn
import numpy as np
import pandas as pd
from plot import Plot
import torch.optim as optim
from getData import getData
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pca_process import transformation_data
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from dgl.sampling import global_uniform_negative_sampling


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(methods, transform_method):
    all_losses = []
    all_accuracys = []
    all_stages = []
    all_methods = []
    for method in methods:
        not_continue = False
        data = getData()

        if method == '':
            method = 'seq_to_seq'
            name = 'DB: PIP y features ' + method
            g = data.seq_to_seq_edge_regression()
            if transform_method == 'pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_pca_linear(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca).to(device)
            elif transform_method == 'kernel_pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_kernel_pca(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca).to(device)
            elif transform_method == 'tsne':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_tsne(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca).to(device)
        else:
            name = 'DB: PIP y features ' + method
            g = data.encoding_edge_regression(method)
            if transform_method == 'pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_pca_linear(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca).to(device)
            elif transform_method == 'kernel_pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_kernel_pca(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca).to(device)
            elif transform_method == 'tsne':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_tsne(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca).to(device)
        num_features = g.ndata['feat'][0].shape[0]
        num_edges = g.number_of_edges()

        print("Utilizando metodo ", method)

        n = int(num_edges * 0.1) # Primera division  
        print("Se utilizara solo " + str(n) + " aristas en vez de las " + str(num_edges) + " del grafo")
        ids_edges = list(range(num_edges))

        # Especificar el tamaño de la muestra de prueba
        tamano_val = int(n * 0.4) # Aquí se define la division entre el set de entrenamiento y validacion, (20% - 60%)
        tamano_test = int(n * 0.2) # Aquí se reserva un 20% de la muestra para la validacion
        # Dividir aleatoriamente la lista de edges
        random.seed(123)
        random.shuffle(ids_edges)
        set_entrenamiento = ids_edges[tamano_val:n]
        set_val = ids_edges[tamano_test:tamano_val]
        set_test = ids_edges[:tamano_test]

        sub_g = dgl.edge_subgraph(g, set_entrenamiento)
        sub_g_val = dgl.edge_subgraph(g, set_val)
        sub_g_test = dgl.edge_subgraph(g, set_test)

        # SET ENTRENAMIENTO
        src, dst = sub_g.edges()
        # concatenar los tensores a lo largo de la dimensión 0
        train_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

        # Concatenar node features
        # Obtener los índices de los nodos conectados
        src_idx = train_edges[0]  # Nodo origen
        dst_idx = train_edges[1]  # Nodo destino
        # Obtener las características de los nodos conectados
        src_features = sub_g.ndata['feat'][src_idx]  # Características del nodo origen
        dst_features = sub_g.ndata['feat'][dst_idx]  # Características del nodo destino
        # Cambiar la forma de los tensores a (num_edges, num_features)
        src_features = src_features.view(-1, sub_g.ndata['feat'].shape[1])
        dst_features = dst_features.view(-1, sub_g.ndata['feat'].shape[1])
        # Concatenar las características de los nodos en un nuevo tensor
        train_concat_features = torch.cat((src_features, dst_features), dim=1)

        # SET VALIDACION
        src, dst = sub_g_val.edges()
        # concatenar los tensores a lo largo de la dimensión 0
        val_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

        # Concatenar node features
        # Obtener los índices de los nodos conectados
        src_idx = val_edges[0]  # Nodo origen
        dst_idx = val_edges[1]  # Nodo destino
        # Obtener las características de los nodos conectados
        src_features = sub_g_val.ndata['feat'][src_idx]  # Características del nodo origen
        dst_features = sub_g_val.ndata['feat'][dst_idx]  # Características del nodo destino
        # Cambiar la forma de los tensores a (num_edges, num_features)
        src_features = src_features.view(-1, sub_g_val.ndata['feat'].shape[1])
        dst_features = dst_features.view(-1, sub_g_val.ndata['feat'].shape[1])
        # Concatenar las características de los nodos en un nuevo tensor
        val_concat_features = torch.cat((src_features, dst_features), dim=1)

        # SET TEST
        src, dst = sub_g_test.edges()
        # concatenar los tensores a lo largo de la dimensión 0
        test_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

        # Concatenar node features
        # Obtener los índices de los nodos conectados
        src_idx = test_edges[0]  # Nodo origen
        dst_idx = test_edges[1]  # Nodo destino
        # Obtener las características de los nodos conectados
        src_features = sub_g_test.ndata['feat'][src_idx]  # Características del nodo origen
        dst_features = sub_g_test.ndata['feat'][dst_idx]  # Características del nodo destino
        # Cambiar la forma de los tensores a (num_edges, num_features)
        src_features = src_features.view(-1, sub_g_test.ndata['feat'].shape[1])
        dst_features = dst_features.view(-1, sub_g_test.ndata['feat'].shape[1])
        # Concatenar las características de los nodos en un nuevo tensor
        test_concat_features = torch.cat((src_features, dst_features), dim=1)

        # Ciclo de entrenamiento
        train_loss = []
        val_loss = []

        train_concat_features = train_concat_features.numpy()
        val_concat_features = val_concat_features.numpy()
        test_concat_features = test_concat_features.numpy()

        # Modelo
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        model.fit(train_concat_features, sub_g.edata['weight'].numpy())

        # Train
        train_pred = model.predict(train_concat_features)
        loss = mean_squared_error(sub_g.edata['weight'].numpy(), train_pred)
        print('Train loss: ', loss)
        all_losses.append(loss)
        all_stages.append('train')
        all_methods.append(method)

        # Val
        val_pred = model.predict(val_concat_features)
        loss = mean_squared_error(sub_g_val.edata['weight'].numpy(), val_pred)
        print('Val loss: ', loss)
        all_losses.append(loss)
        all_stages.append('validate')
        all_methods.append(method)

        # Test
        test_pred = model.predict(test_concat_features)
        loss = mean_squared_error(sub_g_test.edata['weight'].numpy(), test_pred)
        print('Test loss: ', loss)
        all_losses.append(loss)
        all_stages.append('test')
        all_methods.append(method)
    tabla = pd.DataFrame()
    tabla['Caracteristicas'] = all_methods
    tabla['Etapa'] = all_stages
    tabla['Perdida(MSE)'] = all_losses
    tabla.to_csv('tables_random_forest/' + transform_method + '/pip_edge_regression_table.csv', index=False)

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     methods = ['']
    # else:
    #     methods = [sys.argv[1]]
    ruta = 'Protein-Protein/encoded_sequences'
    methods = ['']
    for elemento in os.listdir(ruta):
        if os.path.isdir(os.path.join(ruta, elemento)):
            methods.append(elemento)
    transform_method = ['no_transform', 'pca', 'kernel_pca']
    for i in transform_method:
        main(methods, i)

import sys
import dgl
import os
import torch
import random
import torch.nn as nn
import numpy as np
import pandas as pd
from plot import Plot
import torch.optim as optim
from getData import getData
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pca_process import transformation_data
from keras.losses import binary_crossentropy
from cnn_architectures import Models
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
            g = data.seq_to_seq_linkpred()
            if transform_method == 'pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_pca_linear(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca)
            elif transform_method == 'kernel_pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_kernel_pca(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca)
            elif transform_method == 'tsne':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_tsne(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca)
        else:
            name = 'DB: PIP y features ' + method
            g = data.encoding_link_pred(method)
            if transform_method == 'pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_pca_linear(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca)
            elif transform_method == 'kernel_pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_kernel_pca(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca)
            elif transform_method == 'tsne':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_tsne(g.ndata['feat'])
                g.ndata['feat'] = torch.Tensor(data_transform_pca)
        num_features = g.ndata['feat'][0].shape[0]
        num_edges = g.number_of_edges()

        print("Utilizando metodo ", method)

        n = int(num_edges * 0.4) # Primera division  
        print("Se utilizara solo " + str(n) + " aristas en vez de las " + str(num_edges) + " del grafo")
        ids_edges = list(range(num_edges))

        # Especificar el tamaño de la muestra de prueba
        tamano_test = int(n * 0.2) # Aquí se reserva un 20% de la muestra para la validacion
        # Dividir aleatoriamente la lista de edges
        random.seed(123)
        random.shuffle(ids_edges)
        set_entrenamiento = ids_edges[tamano_test:n]
        set_test = ids_edges[:tamano_test]

        sub_g = dgl.edge_subgraph(g, set_entrenamiento)
        sub_g_test = dgl.edge_subgraph(g, set_test)

        # SET ENTRENAMIENTO
        src, dst = sub_g.edges()
        # concatenar los tensores a lo largo de la dimensión 0
        train_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)
        train_edges_label = torch.ones(train_edges.size()[1])

        torch.manual_seed(40)
        # # Muestras negativas ENTRENAMIENTO
        neg_train_edges = global_uniform_negative_sampling(sub_g, sub_g.num_edges()//4)
        src, dst = neg_train_edges
        # concatenar los tensores a lo largo de la dimensión 0
        neg_train_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)
        neg_train_edges_label = torch.zeros(train_edges.size()[1])

        train_edge_index = torch.cat([train_edges, neg_train_edges], dim=1)
        train_y = torch.cat([train_edges_label, neg_train_edges_label], dim=0)

        torch.manual_seed(42)
        idx = torch.randperm(train_edge_index.size()[1])

        train_edge_index = train_edge_index[:, idx].contiguous()
        train_y = train_y[idx].contiguous()

        # Concatenar node features
        # Obtener los índices de los nodos conectados
        src_idx = train_edge_index[0]  # Nodo origen
        dst_idx = train_edge_index[1]  # Nodo destino
        # Obtener las características de los nodos conectados
        src_features = sub_g.ndata['feat'][src_idx]  # Características del nodo origen
        dst_features = sub_g.ndata['feat'][dst_idx]  # Características del nodo destino
        # Cambiar la forma de los tensores a (num_edges, num_features)
        src_features = src_features.view(-1, sub_g.ndata['feat'].shape[1])
        dst_features = dst_features.view(-1, sub_g.ndata['feat'].shape[1])
        # Concatenar las características de los nodos en un nuevo tensor
        train_concat_features = torch.cat((src_features, dst_features), dim=1)

        # SET TEST
        src, dst = sub_g_test.edges()
        # concatenar los tensores a lo largo de la dimensión 0
        test_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)
        test_edges_label = torch.ones(test_edges.size()[1])

        # # Muestras negativas TESTEO
        neg_test_edges = global_uniform_negative_sampling(sub_g_test, sub_g_test.num_edges())
        src, dst = neg_test_edges
        # concatenar los tensores a lo largo de la dimensión 0
        neg_test_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)
        neg_test_edges_label = torch.zeros(test_edges.size()[1])

        test_edge_index = torch.cat([test_edges, neg_test_edges], dim=1)
        test_y = torch.cat([test_edges_label, neg_test_edges_label], dim=0)

        torch.manual_seed(50)
        idx = torch.randperm(test_edge_index.size()[1])

        test_edge_index = test_edge_index[:, idx].contiguous()
        test_y = test_y[idx].contiguous()

        # Concatenar node features
        # Obtener los índices de los nodos conectados
        src_idx = test_edge_index[0]  # Nodo origen
        dst_idx = test_edge_index[1]  # Nodo destino
        # Obtener las características de los nodos conectados
        src_features = sub_g_test.ndata['feat'][src_idx]  # Características del nodo origen
        dst_features = sub_g_test.ndata['feat'][dst_idx]  # Características del nodo destino
        # Cambiar la forma de los tensores a (num_edges, num_features)
        src_features = src_features.view(-1, sub_g_test.ndata['feat'].shape[1])
        dst_features = dst_features.view(-1, sub_g_test.ndata['feat'].shape[1])
        # Concatenar las características de los nodos en un nuevo tensor
        test_concat_features = torch.cat((src_features, dst_features), dim=1)

        train_concat_features = train_concat_features.squeeze().numpy()
        test_concat_features = test_concat_features.squeeze().numpy()
        train_y = train_y.numpy()
        test_y = test_y.numpy()

         # Modelo
        number_epochs = 200
        models = Models(train_concat_features, train_y, test_concat_features, test_y, [0, 1], "A")
        models.fit_models(number_epochs, 1)
        metrics = models.get_metrics()
        print(metrics)
    # tabla = pd.DataFrame()
    # tabla['Caracteristicas'] = all_methods
    # tabla['Etapa'] = all_stages
    # tabla['Precision'] = all_accuracys
    # tabla['Perdida(BCE)'] = all_losses
    # tabla.to_csv('tables_random_forest/' + transform_method + '/pip_linkpred_table.csv', index=False)

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
    transform_method = ['no_transform']
    methods = ['']
    for i in transform_method:
        main(methods, i)

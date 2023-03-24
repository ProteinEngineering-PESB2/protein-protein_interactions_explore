import os
import torch
import pandas as pd
import numpy as np
from getData import getData
from pca_process import transformation_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(methods, transform_method):
    all_losses = []
    all_stages = []
    all_methods = []
    for method in methods:
        data = getData()

        print('Procesando metodo: ', method)
        if method == '':
            method = 'seq_to_seq'
            name = 'DB: PIP y features ' + method
            g = data.seq_to_seq_edge_regression()
            if transform_method == 'pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_pca_linear(g.ndata['feat'])
                data_transform_pca = np.round(data_transform_pca, 3)
                g.ndata['feat'] = torch.Tensor(data_transform_pca)
            elif transform_method == 'kernel_pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_kernel_pca(g.ndata['feat'])
                data_transform_pca = np.round(data_transform_pca, 3)
                g.ndata['feat'] = torch.Tensor(data_transform_pca)
        else:
            name = 'DB: PIP y features ' + method
            g = data.encoding_edge_regression(method)
            if transform_method == 'pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_pca_linear(g.ndata['feat'])
                data_transform_pca = np.round(data_transform_pca, 3)
                g.ndata['feat'] = torch.Tensor(data_transform_pca)
            elif transform_method == 'kernel_pca':
                transform_instance = transformation_data()
                data_transform_pca, pca_instance = transform_instance.apply_kernel_pca(g.ndata['feat'])
                data_transform_pca = np.round(data_transform_pca, 3)
                g.ndata['feat'] = torch.Tensor(data_transform_pca)

        # Edges
        src, dst = g.edges()
        # concatenar los tensores a lo largo de la dimensión 0
        train_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

        # Concatenar node features
        # Obtener los índices de los nodos conectados
        src_idx = train_edges[0]  # Nodo origen
        dst_idx = train_edges[1]  # Nodo destino
        # Obtener las características de los nodos conectados
        src_features = g.ndata['feat'][src_idx]  # Características del nodo origen
        dst_features = g.ndata['feat'][dst_idx]  # Características del nodo destino
        # Cambiar la forma de los tensores a (num_edges, num_features)
        src_features = src_features.view(-1, g.ndata['feat'].shape[1])
        dst_features = dst_features.view(-1, g.ndata['feat'].shape[1])
        # Concatenar las características de los nodos en un nuevo tensor
        train_concat_features = torch.cat((src_features, dst_features), dim=1)

        ruta = 'pip_data_modelos/concat_graph/' + transform_method + '/' + method
        if not os.path.exists(ruta):
            os.mkdir(ruta)
        
        concat_features = pd.DataFrame(np.round(train_concat_features.numpy(), 3))
        concat_features.to_csv('pip_data_modelos/concat_graph/' + transform_method + '/' + method + '/concat_features.csv', index=False)
        
        if not os.path.isfile('pip_data_modelos/concat_graph/labels.csv'):
            print('Se crea edge_labels')
            edge_labels = pd.DataFrame()
            edge_labels['labels'] = g.edata['weight'].tolist()
            edge_labels.to_csv('pip_data_modelos/concat_graph/labels.csv', index=False)

if __name__ == "__main__":
    ruta = 'Protein-Protein/encoded_sequences'
    methods = ['']
    
    for elemento in os.listdir(ruta):
        if os.path.isdir(os.path.join(ruta, elemento)):
            methods.append(elemento)

    transform_method = ['no_transform', 'pca', 'kernel_pca']
    methods = ['']
    for i in transform_method:
        main(methods, i)

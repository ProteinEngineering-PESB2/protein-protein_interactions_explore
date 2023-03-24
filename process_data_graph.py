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

        ruta = 'pip_data_modelos/' + transform_method + '/' + method
        if not os.path.exists(ruta):
            os.mkdir(ruta)
        
        nodes_features = pd.DataFrame(np.round(g.ndata['feat'].numpy(), 3))
        nodes_features.to_csv('pip_data_modelos/' + transform_method + '/' + method + '/nodes_features_.csv', index=False)
        
        if not os.path.isfile('pip_data_modelos/edge_index.csv'):
            print('Se crea edge_index')
            edge_index = pd.DataFrame()
            src, dst = g.edges()
            train_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)
            edge_index['edge_1'] = train_edges.tolist()[0]
            edge_index['edge_2'] = train_edges.tolist()[1]
            edge_index.to_csv('pip_data_modelos/edge_index.csv', index=False)
        
        if not os.path.isfile('pip_data_modelos/weights.csv'):
            print('Se crea edge_weights')
            edge_weights = pd.DataFrame()
            edge_weights['weights'] = g.edata['weight'].tolist()
            edge_weights.to_csv('pip_data_modelos/weights.csv', index=False)

        if not os.path.isfile('pip_data_modelos/labels.csv'):
            print('Se crea edge_labels')
            edge_labels = pd.DataFrame()
            edge_labels['labels'] = g.edata['label'].tolist()
            edge_labels.to_csv('pip_data_modelos/labels.csv', index=False)


if __name__ == "__main__":
    ruta = 'Protein-Protein/encoded_sequences'
    methods = ['']
    
    for elemento in os.listdir(ruta):
        if os.path.isdir(os.path.join(ruta, elemento)):
            methods.append(elemento)

    transform_method = ['no_transform', 'pca', 'kernel_pca']
    for i in transform_method:
        main(methods, i)

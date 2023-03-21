import sys
import os
import torch
import dgl
import random
import pandas as pd
from plot import Plot
import torch.nn as nn
import torch.optim as optim
from getData import getData
from model_pip import EdgeWeightPredictorMLP
from pca_process import transformation_data


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(methods, transform_method):
    all_losses = []
    all_stages = []
    all_methods = []
    for method in methods:
        not_continue = False
        data = getData()

        if transform_method == 'tsne':
            if method not in ['oneshot', 'bepler', 'glove', 'fasttext']:
                not_continue = True
        if not not_continue:
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

            n = int(num_edges * 0.4) # Primera division  
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

            # SET VALIDACION
            src, dst = sub_g_val.edges()
            # concatenar los tensores a lo largo de la dimensión 0
            val_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

            # SET TEST
            src, dst = sub_g_test.edges()
            # concatenar los tensores a lo largo de la dimensión 0
            test_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

            model = EdgeWeightPredictorMLP(num_features, 1)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)

            train_loss = []
            val_loss = []
            # Entrenar el modelo
            num_epochs = 50
            for epoch in range(num_epochs):
                model.train()
                optimizer.zero_grad()
                out = model(sub_g.ndata['feat'], train_edges)
                loss = criterion(out, sub_g.edata['weight'])
                loss.backward()
                optimizer.step()
                # print(f"-----------------------------------------Epoch {epoch}") 
                # print(f"Train loss: {loss.item():.4f}")
                train_loss.append(loss.item())
                if epoch == num_epochs - 1:
                    all_losses.append(loss.item())
                    all_stages.append('train')
                    all_methods.append(method)
                with torch.no_grad():
                    model.eval()
                    out = model(sub_g_val.ndata['feat'], val_edges)
                    loss = criterion(out, sub_g_val.edata['weight'])
                    # print(f"Validation loss: {loss.item():.4f}")
                    val_loss.append(loss.item())
                    if epoch == num_epochs - 1:
                        all_losses.append(loss.item())
                        all_stages.append('validate')
                        all_methods.append(method)
            # Testeo
            with torch.no_grad():
                model.eval()
                out = model(sub_g_test.ndata['feat'], test_edges)
                loss = criterion(out, sub_g_test.edata['weight'])
                # print('------------------TEST--------------------')
                # print(f"Test Loss: {loss.item():.4f}")
                # print("Valor minimo de true edge label: ", sub_g_test.edata['weight'].min())
                # print("Valor maximo de true edge label: ", sub_g_test.edata['weight'].max())
                # print("True edge label: ", sub_g_test.edata['weight'])
                # print("Pred edge label: ", out)
                all_losses.append(loss.item())
                all_stages.append('test')
                all_methods.append(method)
            # Ploter
            plot = Plot()
            plot.graficar_loss_no_cutoff(train_loss, val_loss, name, method, 'edge_regression', num_epochs, transform_method, 'mlp')
            # Guardar modelo
            # print("Entrenamiento termiando y modelo guardandose")
            # torch.save(model, 'modelos/trainned_model_edge_regression_pip.pth')
    tabla = pd.DataFrame()
    tabla['Caracteristicas'] = all_methods
    tabla['Etapa'] = all_stages
    tabla['Perdida(MSE)'] = all_losses
    tabla.to_csv('tables_mlp/' + transform_method + '/pip_edge_regression_table' + str(num_epochs) + '.csv', index=False)

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
    # transform_method = ['tsne']
    for i in transform_method:
        main(methods, i)

import sys
import dgl
import os
import torch
import random
import numpy as np
import pandas as pd
from plot import Plot
from model_pip import Net
from getData import getData
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from pca_process import transformation_data
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

        model = Net(num_features, num_features//2, num_features//2)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        criterion = torch.nn.BCEWithLogitsLoss()

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

        # SET VALIDACION
        src, dst = sub_g_val.edges()
        # concatenar los tensores a lo largo de la dimensión 0
        val_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)
        val_edges_label = torch.ones(val_edges.size()[1])

        # # Muestras negativas VALIDACION
        neg_val_edges = global_uniform_negative_sampling(sub_g_val, sub_g_val.num_edges()//4)
        src, dst = neg_val_edges
        # concatenar los tensores a lo largo de la dimensión 0
        neg_val_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)
        neg_val_edges_label = torch.zeros(val_edges.size()[1])

        val_edge_index = torch.cat([val_edges, neg_val_edges], dim=1)
        val_y = torch.cat([val_edges_label, neg_val_edges_label], dim=0)

        torch.manual_seed(50)
        idx = torch.randperm(val_edge_index.size()[1])

        val_edge_index = val_edge_index[:, idx].contiguous()
        val_y = val_y[idx].contiguous()

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

        # Ciclo de entrenamiento
        train_loss = []
        train_accuracys = []
        val_loss = []
        val_accuracys = []
        epochs = []
        # Numero de epocas
        num_epochs = 200
        cutoff = 0.865739974975586 # 1 o 0.865739974975586 o tambien la mediana de algo
        for epoch in range(num_epochs):
            epochs.append(epoch + 1)
            model.train()
            optimizer.zero_grad()
            z = model.encode(sub_g.ndata['feat'], train_edges)
            out, binary = model.decode(z, train_edge_index, cutoff)
            accuracy = (binary == train_y).float().mean().item()
            loss = criterion(out, train_y)
            loss.backward()
            # print(f'-----------------------------------------Epoch: {epoch:03d}')
            loss_binary = F.binary_cross_entropy_with_logits(binary, train_y.float(), reduction='mean')
            # print('Loss train: ', loss_binary.item())
            # print('Accuracy train: ', accuracy)
            train_loss.append(loss_binary.item())
            train_accuracys.append(accuracy)
            if epoch == num_epochs - 1:
                all_losses.append(loss_binary.item())
                all_accuracys.append(accuracy)
                all_stages.append('train')
                all_methods.append(method)
            optimizer.step()
            # Validación del modelo en un conjunto de validación
            with torch.no_grad():
                model.eval()
                z = model.encode(sub_g_val.ndata['feat'], val_edges)
                out, val_binary = model.decode(z, val_edge_index, cutoff)
                val_accuracy = (val_binary == val_y).float().mean().item()
                loss_binary = F.binary_cross_entropy_with_logits(val_binary, val_y.float(), reduction='mean')
                # print('Loss val: ', loss_binary.item())
                # print('Accuracy val: ', val_accuracy)
                # print('AUC val: ', roc_auc_score(val_y.cpu().numpy(), val_binary.cpu().numpy()))
                val_loss.append(loss_binary.item())
                val_accuracys.append(val_accuracy)
                if epoch == num_epochs - 1:
                    all_losses.append(loss_binary.item())
                    all_accuracys.append(val_accuracy)
                    all_stages.append('validate')
                    all_methods.append(method)
        # Testeo
        with torch.no_grad():
            model.eval()
            z = model.encode(sub_g_test.ndata['feat'], test_edges)
            out, test_binary = model.decode(z, test_edge_index, cutoff)
            loss_binary = F.binary_cross_entropy_with_logits(test_binary, test_y.float(), reduction='mean')
            test_accuracy = (test_binary == test_y).float().mean().item()
            # print('------------------TEST--------------------')
            # print('Loss test: ', loss_binary.item())
            # print('Accuracy test: ', test_accuracy)
            # print('Edges predichos:')
            # print(test_edges)
            # print('Edges nuevos:')
            # print(test_binary)
            all_losses.append(loss_binary.item())
            all_accuracys.append(test_accuracy)
            all_stages.append('test')
            all_methods.append(method)
        # Ploter
        plot = Plot()
        plot.graficar_loss(train_loss, val_loss, cutoff, name, method, 'linkpred', num_epochs, transform_method, 'gnn')
        plot.graficar_accuracy(train_accuracys, val_accuracys, cutoff, name, method, 'linkpred', num_epochs, transform_method, 'gnn')
        # Guardar modelo
        # print('Entrenamiento termiando y modelo guardandose')
        # torch.save(model, 'modelos/trainned_model_linkpred_pip.pth')
    tabla = pd.DataFrame()
    tabla['Caracteristicas'] = all_methods
    tabla['Etapa'] = all_stages
    tabla['Precision'] = all_accuracys
    tabla['Perdida(BCE)'] = all_losses
    tabla.to_csv('tables_gnn/' + transform_method + '/pip_linkpred_table' + str(num_epochs) + '.csv', index=False)

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

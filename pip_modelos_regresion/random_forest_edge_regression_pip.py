import sys
import dgl
import os
import torch
import random
import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


def main(feature, transform, estimators, type_model, test_split):
    Dataframe = pd.read_csv('../pip_data_modelos/edge_index.csv', na_filter = False, dtype={'edge_1': int, 'edge_2': int})
    edge_1 = Dataframe['edge_1'].values
    edge_2 = Dataframe['edge_2'].values
    Dataframe = pd.read_csv('../pip_data_modelos/weights.csv', na_filter = False, dtype={'weights': float})
    weights = Dataframe['weights'].values
    if transform is None:
        transform = 'no_transform'
        Dataframe = pd.read_csv('../pip_data_modelos/no_transform/' + feature + '/nodes_features_.csv', na_filter = False, sep='|')
    elif transform == 'pca':
        Dataframe = pd.read_csv('../pip_data_modelos/pca/' + feature + '/nodes_features_.csv', na_filter = False, sep='|')
    else:
        Dataframe = pd.read_csv('../pip_data_modelos/kernel_pca/' + feature + '/nodes_features_.csv', na_filter = False, sep='|')
    nodes_features_df = Dataframe.values
    nodes_features = []
    for i, col in enumerate(nodes_features_df):
            array = []
            split = nodes_features_df[i][0].split(',')
            for j in split:
                array.append(float(j))
            nodes_features.append(array)

    nodes_features = torch.tensor(nodes_features)
    weights = torch.from_numpy(weights)

    g = dgl.graph((edge_1, edge_2))
    g.ndata['feat'] = nodes_features
    g.edata['weights'] = weights

    num_features = g.ndata['feat'][0].shape[0]
    num_edges = g.number_of_edges()
    # Dividir set de datos
    n = int(num_edges)
    ids_edges = list(range(num_edges))
    tamano_test = int(n * test_split) # Aquí se reserva un 20% de la muestra para el testeo
    # Dividir aleatoriamente la lista de edges
    random.seed(123)
    random.shuffle(ids_edges)
    set_entrenamiento = ids_edges[tamano_test:n]
    set_test = ids_edges[:tamano_test]

    sub_g = dgl.edge_subgraph(g, set_entrenamiento)
    sub_g_test = dgl.edge_subgraph(g, set_test)

    # SET ENTRENAMIENTO
    src, dst = sub_g.edges()
    train_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

    # Concatenar node features
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

    # SET TEST
    src, dst = sub_g_test.edges()
    test_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

    # Concatenar node features
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

    train_concat_features = train_concat_features.numpy()
    test_concat_features = test_concat_features.numpy()
    train_weights = sub_g.edata['weights'].numpy()
    test_weights = sub_g_test.edata['weights'].numpy()

    # Modelo
    model = RandomForestRegressor(n_estimators=estimators, random_state=42, n_jobs=-1)

    df_losses = []
    df_stages = []
    df_methods = []
    df_tranform = []
    df_estimators = []

    print('Iniciando entrenamiento...')
    # Train
    model.fit(train_concat_features, train_weights)
    train_pred = model.predict(train_concat_features)
    loss = mean_squared_error(train_weights, train_pred)
    # print('Train loss: ', loss)
    df_losses.append(loss)
    df_stages.append('train')
    df_methods.append(feature)
    df_tranform.append(transform)
    df_estimators.append(estimators)
    # Test
    test_pred = model.predict(test_concat_features)
    loss = mean_squared_error(test_weights, test_pred)
    # print('Test loss: ', loss)
    df_losses.append(loss)
    df_stages.append('test')
    df_methods.append(feature)
    df_tranform.append(transform)
    df_estimators.append(estimators)
    # print('Entrenamiento termiando y modelo guardandose')
    # torch.save(model, 'modelos/trainned_model_edge_regressor_pip.pth')
    table = pd.DataFrame()
    table['Estimators'] = df_estimators
    table['Features_used'] = df_methods
    table['Transform_method'] = df_tranform
    table['Stage'] = df_stages
    table['MSE'] = df_losses

    print('Entrenamiento finalizado...')
    print('Resultados guardados en resultados/' + type_model + '_edge_regressor_estimators_' + str(estimators) + '_test_' + str(test_split) + '.csv')
    ruta = 'resultados/'
    if not os.path.exists(ruta):
        os.mkdir(ruta)
    table.to_csv('resultados/' + type_model + '_edge_regressor_estimators_' + str(estimators) + '_test_' + str(test_split) + '.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser(usage="(example) random_forest_edge_regressor_pip.py --features bepler --transform(optional) pca --estimators 20 --test 0.3")
    parser.add_argument("--features", metavar="[features to use]", help="Features: [seq_to_seq], NLP: [bepler, fasttext, glove], Onehot: [onehot], FFT: [fft_alpha_structure, fft_betha_structure, fft_energetic, fft_hydropathy, fft_hydrophobicity, fft_index, fft_secondary_structure, fft_volume], Physicochemical properties: [physicochemical_alpha_structure, physicochemical_betha_structure, physicochemical_energetic, physicochemical_hydropathy, physicochemical_hydrophobicity, physicochemical_index, physicochemical_secondary_structure, physicochemical_volume]")
    parser.add_argument("--transform", metavar="[transform method] (optional)", help="PCA: [pca], Kernel-PCA: [kernel_pca]")
    parser.add_argument("--estimators", metavar="[num estimators]", help="Number of estimators")
    parser.add_argument("--test", metavar="[percentage of data for training]", help="Number: [0, 1]")
    args = parser.parse_args()

    if args.features is None or args.test is None or args.estimators is None:
        print("Missing parameters")
        print()
        parser.print_help()
        sys.exit()

    methods = ['seq_to_seq', 'bepler', 'fasttext', 'glove', 'onehot', 'fft_alpha_structure', 'fft_betha_structure', 'fft_energetic', 'fft_hydropathy', 'fft_hydrophobicity', 'fft_index', 'fft_secondary_structure', 'fft_volume', 'physicochemical_alpha_structure', 'physicochemical_betha_structure', 'physicochemical_energetic', 'physicochemical_hydropathy', 'physicochemical_hydrophobicity', 'physicochemical_index', 'physicochemical_secondary_structure', 'physicochemical_volume']
    transform_methods = ['pca', 'kernel_pca']

    if  args.features not in methods:
        print("Features to use invalid, use options below")
        print()
        parser.print_help()
        sys.exit()

    if not args.transform is None:
        if  args.transform not in transform_methods:
            print("Tranform method invalid, use options below")
            print()
            parser.print_help()
            sys.exit()

    try: 
        args.estimators = int(args.estimators)
    except:
        print("Estimators must be an integer")
        print()
        parser.print_help()
        sys.exit()

    if  args.estimators < 1:
        print("Estimators must be greater than 0")
        print()
        parser.print_help()
        sys.exit()

    try: 
        args.test = float(args.test)
    except:
        print("Percentage for test data must be a number between 0-1")
        print()
        parser.print_help()
        sys.exit()

    if  args.test >= 1 or args.test <= 0:
        print("Percentage for test data must be a number between 0-1")
        print()
        parser.print_help()
        sys.exit()

    if args.transform is None:
        main(args.features, None, args.estimators, 'random_forest', args.test)
    else:
        main(args.features, args.transform, args.estimators,  'random_forest', args.test)

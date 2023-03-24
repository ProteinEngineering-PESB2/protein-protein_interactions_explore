import sys
import os
import dgl
import torch
import random
import argparse
import pandas as pd
from models_pip import Net
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(feature, transform, epochs, type_model, test_split, hidden_layers):
    Dataframe = pd.read_csv('../pip_data_modelos/edge_index.csv', na_filter = False, dtype={'edge_1': int, 'edge_2': int})
    edge_1 = Dataframe['edge_1'].values
    edge_2 = Dataframe['edge_2'].values
    Dataframe = pd.read_csv('../pip_data_modelos/labels.csv', na_filter = False, dtype={'labels': int})
    labels = Dataframe['labels'].values
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

    df_losse = []
    df_accuracy = []
    df_stage = []
    df_method = []
    df_transform = []
    df_epochs = []
    
    nodes_features = torch.tensor(nodes_features)
    labels = torch.from_numpy(labels)

    g = dgl.graph((edge_1, edge_2))
    g.ndata['feat'] = nodes_features
    g.edata['label'] = labels

    num_features = g.ndata['feat'][0].shape[0]
    num_edges = g.number_of_edges()

    # Definir hidden_layers si no se declararon
    if hidden_layers == '':
        hidden_layers = num_features

    # Dividir set de datos
    n = int(num_edges)
    ids_edges = list(range(num_edges))
    tamano_test = int(n * test_split) # Aquí se reserva un 20% de la muestra para la validacion
    # Dividir aleatoriamente la lista de edges
    random.seed(123)
    random.shuffle(ids_edges)
    set_entrenamiento = ids_edges[tamano_test:n]
    set_test = ids_edges[:tamano_test]

    sub_g = dgl.edge_subgraph(g, set_entrenamiento).to(device)
    sub_g_test = dgl.edge_subgraph(g, set_test).to(device)
    
    # SET ENTRENAMIENTO
    src, dst = sub_g.edges()
    train_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

    # SET TEST
    src, dst = sub_g_test.edges()
    test_edges = torch.cat([src.unsqueeze(0), dst.unsqueeze(0)], dim=0)

    model = Net(num_features, hidden_layers, 1).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    print('Iniciando entrenamiento...')
    # Ciclo de entrenamiento
    cutoff = 0.5 # 1 o 0.865739974975586
    for epoch in range(epochs):
        df_epochs.append(epoch + 1)
        df_stage.append('train')
        df_method.append(feature)
        df_transform.append(transform)
        model.train()
        optimizer.zero_grad()
        out, binary = model.encode(sub_g.ndata['feat'], train_edges, cutoff)
        accuracy = (binary == sub_g.edata['label']).float().mean().item()
        loss = criterion(out, sub_g.edata['label'].float())
        loss.backward()
        loss_binary = F.binary_cross_entropy_with_logits(binary, sub_g.edata['label'].float(), reduction='mean')
        df_accuracy.append(accuracy)
        df_losse.append(loss_binary.item())
        # print(f'-----------------------------------------Epoch: {epoch:03d}')
        # print('Loss train: ', loss_binary.item())
        # print('Accuracy train: ', accuracy)
        optimizer.step()
        # Testeo
        with torch.no_grad():
            df_epochs.append(epoch + 1)
            df_stage.append('test')
            df_method.append(feature)
            df_transform.append(transform)
            model.eval()
            out, test_binary = model.encode(sub_g_test.ndata['feat'], test_edges, cutoff)
            loss_binary = F.binary_cross_entropy_with_logits(test_binary, sub_g_test.edata['label'].float(), reduction='mean')
            test_accuracy = (test_binary == sub_g_test.edata['label']).float().mean().item()
            df_accuracy.append(test_accuracy)
            df_losse.append(loss_binary.item())
            # print('------------------TEST--------------------')
            # print('Loss test: ', loss_binary.item())
            # print('Accuracy test: ', test_accuracy)
    # print('Entrenamiento termiando y modelo guardandose')
    # torch.save(model, 'modelos/trainned_model_linkpred_pip.pth')
    table = pd.DataFrame()
    table['Epochs'] = df_epochs
    table['Features_used'] = df_method
    table['Transform_method'] = df_transform
    table['Stage'] = df_stage
    table['Accuracy'] = df_accuracy
    table['Loss(BCE)'] = df_losse

    print('Entrenamiento finalizado...')
    print('Resultados guardados en resultados/' + type_model + '_linkpred_epochs_' + str(epochs) + '_test_' + str(test_split) + '_hidden_layers_' + str(hidden_layers) + '.csv')
    ruta = 'resultados/'
    if not os.path.exists(ruta):
        os.mkdir(ruta)
    table.to_csv('resultados/' + type_model + '_linkpred_epochs_' + str(epochs) + '_test_' + str(test_split) + '_hidden_layers_' + str(hidden_layers) + '.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(add_help=False)
    parser = argparse.ArgumentParser(usage="(example) gnn_linkpred_pip.py --features bepler --transform(optional) pca --epochs 200 --test 0.3")
    parser.add_argument("--features", metavar="[features to use]", help="Features: [seq_to_seq], NLP: [bepler, fasttext, glove], Onehot: [onehot], FFT: [fft_alpha_structure, fft_betha_structure, fft_energetic, fft_hydropathy, fft_hydrophobicity, fft_index, fft_secondary_structure, fft_volume], Physicochemical properties: [physicochemical_alpha_structure, physicochemical_betha_structure, physicochemical_energetic, physicochemical_hydropathy, physicochemical_hydrophobicity, physicochemical_index, physicochemical_secondary_structure, physicochemical_volume]")
    parser.add_argument("--transform", metavar="[transform method] (optional)", help="PCA: [pca], Kernel-PCA: [kernel_pca]")
    parser.add_argument("--epochs", metavar="[num epochs]", help="Number of epochs")
    parser.add_argument("--hidden", metavar="[num hidden channels] (optional)", help="Number of hidden channels for GCN/Linear layers")
    parser.add_argument("--test", metavar="[percentage of data for training]", help="Number: [0, 1]")
    args = parser.parse_args()
    if args.features is None or args.epochs is None or args.test is None:
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

    if not args.hidden is None:
        try:
            args.hidden = int(args.hidden)
            if args.hidden < 1:
                print("Hidden channels number must be greater than 0")
                print()
                parser.print_help()
                sys.exit()
        except:
            print("Hidden channels number must be an integer")
            print()
            parser.print_help()
            sys.exit()
    else:
        args.hidden = ''

    try: 
        args.epochs = int(args.epochs)
    except:
        print("Epochs must be an integer")
        print()
        parser.print_help()
        sys.exit()

    if  args.epochs < 1:
        print("Epochs must be greater than 0")
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
        main(args.features, None, args.epochs, 'gnn', args.test, args.hidden)
    else:
        main(args.features, args.transform, args.epochs, 'gnn', args.test, args.hidden)

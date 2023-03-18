import dgl
import torch
import pandas as pd
import torch.nn.functional as F


class getData:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def seq_to_seq_edge_regression(self):
        df = pd.read_csv('Protein-Protein/sequences/unique_sequences.csv', na_filter = False, sep=',',
                                    dtype={'sequence': str, 'id': int})
        sequences = df['sequence'].values
        id_uniq_seq = df['id'].values
        uniq_sequences = {}
        for i, data in enumerate(sequences):
            if sequences[i] not in uniq_sequences:
                uniq_sequences[sequences[i]] = id_uniq_seq[i]
        df = pd.read_csv('Protein-Protein/datasets/pip_database.csv', na_filter = False, sep=',',
                                    dtype={'seq_0': str, 'seq_1': str, 'target': float})
        id1 = df['seq_0'].values
        id2 = df['seq_1'].values
        sc = df['target'].values
        q1 = df['target'].quantile(0.25)
        q3 = df['target'].quantile(0.75)
        iqr = q3 - q1
        id_dic = {}
        #########################
        count = 0
        for i, data in enumerate(id1):
            if id1[i] not in id_dic and id1[i] in uniq_sequences and sc[i] <= q3 + 1.5*iqr:
                id_dic[id1[i]] = count
                count = count + 1
            if id2[i] not in id_dic and id2[i] in uniq_sequences and sc[i] <= q3 + 1.5*iqr:
                id_dic[id2[i]] = count
                count = count + 1
        # EDGE INDEX Y LABEL (TAMBIEN EL SCORE)
        new_id1 = []
        new_id2 = []
        new_score = []
        for i, data in enumerate(id1):
            if id1[i] in uniq_sequences and id2[i] in uniq_sequences and sc[i] <= q3 + 1.5*iqr:    
                new_id1.append(id_dic[id1[i]])
                new_id2.append(id_dic[id2[i]])
                new_score.append(sc[i])
        edge_label = torch.Tensor(new_score).to(self.device)
        # NODES FEATURES
        df_2 = pd.read_csv('features/Seq2Feature_summarized/resultados/result_values_normalized.csv', na_filter = False, header=None, sep='|')
        columna_features = df_2.values
        old_id_score_dic = {}
        num_features = 0
        for i, col in enumerate(columna_features):
            split = columna_features[i][0].split(',')
            id = columna_features[i][0].split(',')[0]
            # features primeras 20 positivas
            # features_to_use = [split[121], split[88], split[87], split[120], split[122],
            #                    split[102], split[65], split[79], split[20], split[53],
            #                    split[63], split[92], split[64], split[108], split[42],
            #                    split[94], split[107], split[67], split[116], split[109]]
            # features primeras 10 positivas y ultimas 10 negativas, de mas negativa a menos
            # features_to_use = [split[121], split[88], split[87], split[120], split[122],
            #                 split[102], split[65], split[79], split[20], split[53],
            #                 split[73], split[112], split[78], split[117], split[85],
            #                 split[22], split[119], split[110], split[118], split[105]]
            # features primeras 8 positivas y ultimas 8 negativas, de mas negativa a menos
            # features_to_use = [split[121], split[88], split[87], split[120], split[122],
            #                    split[102], split[65], split[79], split[20],
            #                    split[73], split[112], split[78], split[117], split[85],
            #                    split[22], split[119], split[110], split[118]]
            # if num_features == 0:
            #     num_features = len(features_to_use)
            old_id_score_dic[id] = [float(x) for x in split[1:]]
        new_score = []
        new_scoreid = []
        for i, data in enumerate(id_dic.keys()):
            new_scoreid.append(data)
            new_score.append(old_id_score_dic[data])
        nodes_features = torch.tensor(new_score).to(self.device)

        g = dgl.graph((new_id1, new_id2))
        g.ndata['feat'] = nodes_features
        edge_label_normalized = F.normalize(edge_label, p=2, dim=0)
        # g.edata['weight'] = edge_label_normalized
        g.edata['weight'] = edge_label

        return g

    def encoding_edge_regression(self, method):
        df = pd.read_csv('Protein-Protein/sequences/unique_sequences.csv', na_filter = False, sep=',',
                                    dtype={'sequence': str, 'id': int})
        sequences = df['sequence'].values
        id_uniq_seq = df['id'].values
        uniq_sequences = {}
        for i, data in enumerate(sequences):
            if sequences[i] not in uniq_sequences:
                uniq_sequences[sequences[i]] = id_uniq_seq[i]
        # NODES FEATURES EXTRACT
        df_2 = pd.read_csv('Protein-Protein/encoded_sequences/' + method + '/dataset_encoding.csv', na_filter = False, sep='|')
        columna_features = df_2.values
        old_id_score_dic = {}
        for i, col in enumerate(columna_features):
            if method[:3] == 'phy' or method[:3] == 'phy':
                split = columna_features[i][0].split(',')
                id = int(float(columna_features[i][0].split(',')[0]))
                old_id_score_dic[id] = [float(x) for x in split[1:]]
            else:
                split = columna_features[i][0].split(',')
                id = int(float(columna_features[i][0].split(',')[-1]))
                old_id_score_dic[id] = [float(x) for x in split[:-1]]
        df = pd.read_csv('Protein-Protein/datasets/pip_database.csv', na_filter = False, sep=',',
                                    dtype={'seq_0': str, 'seq_1': str, 'target': float})
        id1 = df['seq_0'].values
        id2 = df['seq_1'].values
        sc = df['target'].values
        q1 = df['target'].quantile(0.25)
        q3 = df['target'].quantile(0.75)
        iqr = q3 - q1
        #########################
        count = 0
        id_dic = {}
        for i, data in enumerate(id1):
            if id1[i] not in id_dic and id1[i] in uniq_sequences and sc[i] <= q3 + 1.5*iqr:
                id_dic[id1[i]] = count
                count = count + 1
            if id2[i] not in id_dic and id2[i] in uniq_sequences and sc[i] <= q3 + 1.5*iqr:
                id_dic[id2[i]] = count
                count = count + 1
        #########################
        # EDGES
        index_1 = []
        index_2 = []
        new_score = []
        for i, data in enumerate(id1):
            if id1[i] in id_dic and id2[i] in id_dic and sc[i] <= q3 + 1.5*iqr:
                index_1.append(id_dic[id1[i]])
                index_2.append(id_dic[id2[i]])
                new_score.append(sc[i])
        edge_label = torch.tensor(new_score)
        nodes_feat = []
        for i, data in enumerate(id_dic.keys()):
            id_uniq = uniq_sequences[data]
            nodes_feat.append(old_id_score_dic[id_uniq])
        nodes_features = torch.tensor(nodes_feat)

        g = dgl.graph((index_1, index_2))
        g.ndata['feat'] = nodes_features
        g.edata['weight'] = edge_label.float()

        return g

    def seq_to_seq_linkpred(self):
        df = pd.read_csv('Protein-Protein/sequences/unique_sequences.csv', na_filter = False, sep=',',
                                    dtype={'sequence': str, 'id': int})
        sequences = df['sequence'].values
        id_uniq_seq = df['id'].values
        uniq_sequences = {}
        for i, data in enumerate(sequences):
            if sequences[i] not in uniq_sequences:
                uniq_sequences[sequences[i]] = id_uniq_seq[i]
    
        df = pd.read_csv('Protein-Protein/datasets/pip_database.csv', na_filter = False, sep=',',
                                    dtype={'seq_0': str, 'seq_1': str, 'target': float})
        id1 = df['seq_0'].values
        id2 = df['seq_1'].values
        sc = df['target'].values
        id_dic = {}
        ######################### SI SE USA sc[i] >= 1 PARA FILTRAR FUNCIONA BASTANTE BIEN #######################
        count = 0
        for i, data in enumerate(id1):
            if id1[i] not in id_dic and id1[i] in uniq_sequences:
                id_dic[id1[i]] = count
                count = count + 1
            if id2[i] not in id_dic and id2[i] in uniq_sequences:
                id_dic[id2[i]] = count
                count = count + 1
        # EDGE INDEX Y LABEL (TAMBIEN EL SCORE)
        new_id1 = []
        new_id2 = []
        for i, data in enumerate(id1):
            if id1[i] in id_dic and id2[i] in id_dic:
                new_id1.append(id_dic[id1[i]])
                new_id2.append(id_dic[id2[i]])
        # NODES FEATURES
        df_2 = pd.read_csv('features/Seq2Feature_summarized/resultados/result_values_normalized.csv', na_filter = False, header=None, sep='|')
        columna_features = df_2.values
        old_id_score_dic = {}
        num_features = 0
        for i, col in enumerate(columna_features):
            split = columna_features[i][0].split(',')
            id = columna_features[i][0].split(',')[0]
            # features primeras 20 positivas
            # features_to_use = [split[121], split[88], split[87], split[120], split[122],
            #                    split[102], split[65], split[79], split[20], split[53],
            #                    split[63], split[92], split[64], split[108], split[42],
            #                    split[94], split[107], split[67], split[116], split[109]]
            # features primeras 10 positivas y ultimas 10 negativas, de mas negativa a menos
            # features_to_use = [split[121], split[88], split[87], split[120], split[122],
            #                 split[102], split[65], split[79], split[20], split[53],
            #                 split[73], split[112], split[78], split[117], split[85],
            #                 split[22], split[119], split[110], split[118], split[105]]
            # features primeras 8 positivas y ultimas 8 negativas, de mas negativa a menos
            # features_to_use = [split[121], split[88], split[87], split[120], split[122],
            #                    split[102], split[65], split[79], split[20],
            #                    split[73], split[112], split[78], split[117], split[85],
            #                    split[22], split[119], split[110], split[118]]
            # if num_features == 0:
            #     num_features = len(features_to_use)
            old_id_score_dic[id] = [float(x) for x in split[1:]]
        new_score = []
        new_scoreid = []
        for i, data in enumerate(id_dic.keys()):
            new_scoreid.append(data)
            new_score.append(old_id_score_dic[data])
        nodes_features = torch.tensor(new_score).to(self.device)

        g = dgl.graph((new_id1, new_id2))
        g.ndata['feat'] = nodes_features

        return g

    def encoding_link_pred(self, method):
        # NODES FEATURES EXTRACT
        df = pd.read_csv('Protein-Protein/sequences/unique_sequences.csv', na_filter = False, sep=',',
                                    dtype={'sequence': str, 'id': int})
        sequences = df['sequence'].values
        id_uniq_seq = df['id'].values
        uniq_sequences = {}
        for i, data in enumerate(sequences):
            if sequences[i] not in uniq_sequences:
                uniq_sequences[sequences[i]] = id_uniq_seq[i]
        # NODES FEATURES EXTRACT
        df_2 = pd.read_csv('Protein-Protein/encoded_sequences/' + method + '/dataset_encoding.csv', na_filter = False, sep='|')
        columna_features = df_2.values
        old_id_score_dic = {}
        for i, col in enumerate(columna_features):
            if method[:3] == 'phy':
                split = columna_features[i][0].split(',')
                id = int(float(columna_features[i][0].split(',')[0]))
                old_id_score_dic[id] = [float(x) for x in split[1:]]
            else:
                split = columna_features[i][0].split(',')
                id = int(float(columna_features[i][0].split(',')[-1]))
                old_id_score_dic[id] = [float(x) for x in split[:-1]]
        df = pd.read_csv('Protein-Protein/datasets/pip_database.csv', na_filter = False, sep=',',
                                    dtype={'seq_0': str, 'seq_1': str, 'target': float})
        id1 = df['seq_0'].values
        id2 = df['seq_1'].values
        #########################
        count = 0
        id_dic = {}
        for i, data in enumerate(id1):
            if id1[i] not in id_dic and id1[i] in uniq_sequences:
                id_dic[id1[i]] = count
                count = count + 1
            if id2[i] not in id_dic and id2[i] in uniq_sequences:
                id_dic[id2[i]] = count
                count = count + 1
        #########################
        # EDGES
        index_1 = []
        index_2 = []
        for i, data in enumerate(id1):
            if id1[i] in id_dic and id2[i] in id_dic:
                index_1.append(id_dic[id1[i]])
                index_2.append(id_dic[id2[i]])
        nodes_feat = []
        for i, data in enumerate(id_dic.keys()):
            id_uniq = uniq_sequences[data]
            nodes_feat.append(old_id_score_dic[id_uniq])
        nodes_features = torch.tensor(nodes_feat)

        g = dgl.graph((index_1, index_2))
        g.ndata['feat'] = nodes_features

        return g

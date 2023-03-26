import dgl
import torch
import pandas as pd
import torch.nn.functional as F


class getData:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def seq_to_seq(self, database):
        df = pd.read_csv('Protein-Protein/sequences/unique_sequences.csv', na_filter = False, sep=',',
                                    dtype={'sequence': str, 'id': int})
        sequences = df['sequence'].values
        id_uniq_seq = df['id'].values
        uniq_sequences = {}
        for i, data in enumerate(sequences):
            if sequences[i] not in uniq_sequences:
                uniq_sequences[sequences[i]] = id_uniq_seq[i]
        
        # ALL DATABASES
        databases_headers = {'kd_pdbbind_database.csv': ['seq_0', 'seq_1', 'target'],
                             'proximate_dg.csv': ['sequence_0', 'sequence_1', 'dg'],
                             'proximate_kd.csv': ['sequence_0', 'sequence_1', 'kd'],
                             'proximate_kon.csv': ['sequence_0', 'sequence_1', 'kon'],
                             'skempi_affinity.csv': ['sequence_0', 'sequence_1', 'affinity'],
                             'skempi_koff.csv': ['sequence_0', 'sequence_1', 'koff'],
                             'skempi_kon.csv': ['sequence_0', 'sequence_1', 'kon']}
        
        headers = databases_headers[database]
        df = pd.read_csv('Protein-Protein/datasets/' + database, na_filter = False, sep=',',
                                    dtype={headers[0]: str, headers[1]: str, headers[2]: float})
        id1 = df[headers[0]].values
        id2 = df[headers[1]].values
        sc = df[headers[2]].values
        # q1 = df['target'].quantile(0.25)
        # q3 = df['target'].quantile(0.75)
        # iqr = q3 - q1
        count = 0
        id_dic = {}
        for i, data in enumerate(id1):
            if id1[i] not in id_dic and id1[i] in uniq_sequences: # and sc[i] <= q3 + 1.5*iqr:
                id_dic[id1[i]] = count
                count = count + 1
            if id2[i] not in id_dic and id2[i] in uniq_sequences: # and sc[i] <= q3 + 1.5*iqr:
                id_dic[id2[i]] = count
                count = count + 1
        # EDGE INDEX Y LABEL (TAMBIEN EL SCORE)
        new_id1 = []
        new_id2 = []
        new_score = []
        label = []
        for i, data in enumerate(id1):
            if id1[i] in uniq_sequences and id2[i] in uniq_sequences: #  and sc[i] <= q3 + 1.5*iqr:    
                new_id1.append(id_dic[id1[i]])
                new_id2.append(id_dic[id2[i]])
                new_score.append(sc[i])
                label.append(1)
        label = torch.tensor(label)
        edge_label = torch.Tensor(new_score)
        # NODES FEATURES
        df_2 = pd.read_csv('features/Seq2Feature_summarized/resultados/result_values_normalized.csv', na_filter = False, header=None, sep='|')
        columna_features = df_2.values
        old_id_score_dic = {}
        num_features = 0
        for i, col in enumerate(columna_features):
            split = columna_features[i][0].split(',')
            id = columna_features[i][0].split(',')[0]
            old_id_score_dic[id] = [round(float(x), 3) for x in split[1:]]
        new_score = []
        new_scoreid = []
        for i, data in enumerate(id_dic.keys()):
            new_scoreid.append(data)
            new_score.append(old_id_score_dic[data])
        nodes_features = torch.tensor(new_score)

        g = dgl.graph((new_id1, new_id2))
        g.ndata['feat'] = nodes_features
        edge_label_normalized = F.normalize(edge_label, p=2, dim=0)
        # g.edata['weight'] = edge_label_normalized
        g.edata['weight'] = edge_label
        g.edata['label'] = label

        return g

    def encoding(self, method, database):
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
                old_id_score_dic[id] = [round(float(x), 3) for x in split[1:]]
            else:
                split = columna_features[i][0].split(',')
                id = int(float(columna_features[i][0].split(',')[-1]))
                old_id_score_dic[id] = [round(float(x), 3) for x in split[:-1]]
        
        
        # ALL DATABASES
        databases_headers = {'kd_pdbbind_database.csv': ['seq_0', 'seq_1', 'target'],
                             'proximate_dg.csv': ['sequence_0', 'sequence_1', 'dg'],
                             'proximate_kd.csv': ['sequence_0', 'sequence_1', 'kd'],
                             'proximate_kon.csv': ['sequence_0', 'sequence_1', 'kon'],
                             'skempi_affinity.csv': ['sequence_0', 'sequence_1', 'affinity'],
                             'skempi_koff.csv': ['sequence_0', 'sequence_1', 'koff'],
                             'skempi_kon.csv': ['sequence_0', 'sequence_1', 'kon']}
        
        headers = databases_headers[database]
        df = pd.read_csv('Protein-Protein/datasets/' + database, na_filter = False, sep=',',
                                    dtype={headers[0]: str, headers[1]: str, headers[2]: float})
        id1 = df[headers[0]].values
        id2 = df[headers[1]].values
        sc = df[headers[2]].values
        # q1 = df['target'].quantile(0.25)
        # q3 = df['target'].quantile(0.75)
        # iqr = q3 - q1
        count = 0
        id_dic = {}
        for i, data in enumerate(id1):
            if id1[i] not in id_dic and id1[i] in uniq_sequences: #  and sc[i] <= q3 + 1.5*iqr:
                id_dic[id1[i]] = count
                count = count + 1
            if id2[i] not in id_dic and id2[i] in uniq_sequences: #  and sc[i] <= q3 + 1.5*iqr:
                id_dic[id2[i]] = count
                count = count + 1
        #########################
        # EDGES
        index_1 = []
        index_2 = []
        new_score = []
        label = []
        for i, data in enumerate(id1):
            if id1[i] in id_dic and id2[i] in id_dic: #  and sc[i] <= q3 + 1.5*iqr:
                index_1.append(id_dic[id1[i]])
                index_2.append(id_dic[id2[i]])
                new_score.append(sc[i])
                label.append(1)
        label = torch.tensor(label)
        edge_label = torch.tensor(new_score)
        nodes_feat = []
        for i, data in enumerate(id_dic.keys()):
            id_uniq = uniq_sequences[data]
            nodes_feat.append(old_id_score_dic[id_uniq])
        nodes_features = torch.tensor(nodes_feat)

        g = dgl.graph((index_1, index_2))
        g.ndata['feat'] = nodes_features
        g.edata['weight'] = edge_label.float()
        g.edata['label'] = label

        return g

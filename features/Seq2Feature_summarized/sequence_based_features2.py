import pandas as pd
from Bio import SeqIO
import numpy as np
import timeit
from multiprocessing import Process, cpu_count
import multiprocessing


def find_features(seqs, df_dict_noNormalized, df_dict_normalized):
    for i, inter in enumerate(seqs):
        sequence = seqs[i]
        # Reemplazar caracteres no admitidos X o Z? que pasa con las letras B o J
        seq = sequence.strip().split('\t')[0].replace('X', '').replace('Z', '').replace('B', '').replace('J', '').replace('U', '').replace('O', '')

        newDF = pd.DataFrame()
        newDF_1 = pd.DataFrame()

        file1 = pd.read_csv('./data/prop_49_list.csv')
        new = pd.DataFrame()
        new_1 = pd.DataFrame()
        df1 = pd.read_csv('./data/49_properties_numerical_Values.csv', sep=',')
        df_1 = pd.read_csv('./data/49_properties_normalizedValues.csv', sep=',')
        ###############################################
        # DUDOSO
        #
        # no-normalized
        new['Actual'] = df1[list(seq)].mean(axis=1)
        newDF_temp = pd.concat([file1, new], axis=1)
        new = newDF_temp.transpose()
        newDF = newDF.append(new.T, ignore_index=False)
        # normalized
        new_1['Normalized'] = df_1[list(seq)].mean(axis=1)
        newDF_temp_ = pd.concat([file1, new_1], axis=1)
        new_1 = newDF_temp_.transpose()
        newDF_1 = newDF_1.append(new_1.T, ignore_index=False)
        
        newDF = newDF.drop_duplicates(keep='first')
        newDF_1 = newDF_1.drop_duplicates(keep='first')

        array = newDF.iloc[:, 1].values
        array = np.insert(array, 0, sequence)
        df_dict_noNormalized.append(array.tolist())

        array = newDF_1.iloc[:, 1].values
        array = np.insert(array, 0, sequence)
        df_dict_normalized.append(array.tolist())
        #
        ###############################################
        
def main():
    # SABER EL NUMERO DE PROCESOS QUE SE PUEDEN UTILIZAR PARALELAMENTE
    num_workers = cpu_count()
    # TIMER INICIO
    start = timeit.default_timer()
    # LEER FASTA
    print("Empieza a leer las secuencias")
    df = pd.read_csv('unique_sequences.csv', na_filter = False, index_col=None,
                                dtype={'sequence': str, 'id': int})
    sequences = df['sequence'].values
    print("Lectura de secuencias lista")
    print("Cantidad de secuencias ", len(sequences))
    # CONVERTIR EN ARRAY DE NUMPY
    sequences_np = np.array(sequences)
    # DIVIDIR EL ARRAY DE DIP EN PARTES IGUALES
    # sequences_split = np.array_split(sequences_np[:240], num_workers)
    sequences_split = np.array_split(sequences_np, num_workers)
    sequences, sequences_np = None, None
    # MULTIPROCESSING
    procesos = []
    df_dict_noNormalized = multiprocessing.Manager().list()
    df_dict_normalized = multiprocessing.Manager().list()
    print("Informacion separada correctamente")
    print(len(sequences_split[0]))
    for i, splited_data in enumerate(sequences_split):
        proceso = Process(target=find_features, args=(sequences_split[i], df_dict_noNormalized, df_dict_normalized))
        procesos.append(proceso)
        proceso.start()
        print("Iniciado proceso numero: ", i)
    for p in procesos:
        p.join()
    # CREACION DEL DATAFRAME A EXPORTAR
    # header = ['fasta_name','K0','Ht','Hp','P','pHi','pK','Mw','Bl','Rf','Mu','Hnc','Esm','El','Et','Pa','Pb','Pt','Pc','Ca','F','Br','Ra','Ns','aN','aC','aM','V0','Nm','Nl',   'Hgm','ASAD','ASAN','dASA','dGh','GhD','GhN','dHh','-TdSh','dCph','dGc','dHc','-TdSc','dG','dH','-TdS','v','s','f','Pf-s','GEIM800105','GEIM800108','BIOV880102','GRAR740102','GRAR740103','HOPA770101','ISOY800102','ISOY800103','ISOY800104','JOND750101','JUKT750101','KANM800102','KANM800103','KARP850101','KRIW710101','KRIW790101','LAWE840101','LEVM760101','LIFS790101','LIFS790103','MANP780101','MAXF760102','MAXF760106','MIYS850101','NAGK730102','NAGK730103','BURA740102','ARGP820101','NISK860101','OOBM770103','OOBM850103','CHAM820101','OOBM850105','PONP800102','PONP800103','PONP800107','PRAM900104','RACS770101','RACS770102','RACS820113','CHOC760101','ROBB760105','ROSM880102','SIMZ760101','WOEC730101','CHOP780202','YUTK870102','ZIMJ680101','ZIMJ680102','ZIMJ680105','CHOP780204','ONEK900102','VINM940101','NADH010101','NADH010102','NADH010104','NADH010105','FUKS010102','FUKS010103','KUHL950101','ZHOH040101','ZHOH040102','ZHOH040103','PONJ960101','WOLR790101','OLSK800101','KIDA850101','CORJ870102','CORJ870104','CORJ870106','CORJ870108','MIYS990101','FASG890101','DAYM780201','EISD860101','FASG760101','FASG760102','FAUJ830101','FAUJ880101','FAUJ880106','BIGC670101']
    df_noNormlized_export = pd.DataFrame(np.array(df_dict_noNormalized))
    df_Normlized_export = pd.DataFrame(np.array(df_dict_normalized))
    print("Export to csv")
    # MODIFICAR RUTA DE EXPORTACION DEPENDIENDO DE TU MAQUINA
    df_noNormlized_export.to_csv("result_values_noNormalized.csv", index=False)
    df_Normlized_export.to_csv("result_values_normalized.csv", index=False)
    # TIMER FIN
    stop = timeit.default_timer()
    # TIEMPO DE EJECUCION
    execution_time = stop - start
    print("Todo listo, se demoro " + str(execution_time) + " segundos")

if __name__ == "__main__":
    main()

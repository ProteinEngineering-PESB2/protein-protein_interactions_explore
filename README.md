# Protein-protein_interactions_explore

# Requisitos

Para poder correr los modelos es necesario incoporar una caperta llamada "pip_data_modelos" en al carpeta raiz. Esta carpeta contiene los datos de pip transformados en matrices y todas las representaciones disponibles, mas transformaciones  PCA o Kenerl PCA.

Link pip_data_modelos: https://drive.google.com/drive/folders/1uXwGifLdCDT7KPy_n7MILXuFt-paBY33?usp=sharing


# Python - dependecias

Los modelos requieren principalmente 3 librerias para poder funcionar: Pytorch, Pytorch Geometric y DGL

Instalar en orden Pytorch -> Pytorch Geometric -> DGL. De esta manera deberia ser suficiente para tener todos los modulos principales y complementarios para correr los scripts sin problemas.

Es importante tener en cuenta que versiones de CUDA se cuenta ya que estas 3 librerias se instalan de acuerdo a cual esta disponible. 

Hay un "requirements.txt" que contiene todo los modulos necesarios a ser instalados utilizando pip. Para obtener este archivo se hizo un ambiente de cero en python version 3.7.9 y se instalaron las dependecias descritas arriba, en una maquina con CUDA 12.0, donde finalmente se obtuvo la version para CUDA 11.7 en cada uno (DGL solo permite 11.7 y Pytorch geometric solo hasta 11.8).

# Uso

Para todos los scripts dentro de "pip_modelos_clasificacion" o  "pip_modelos_regresion", se puede usar: 

```bash
python nombre_script -h o python nombre_script --help
```
Asi se desplegara una manual de uso con un ejemplo de como se puede ejecutar cada script.

# Parametros

Dentro de los parametros que se utilizan estan:

- **features**: seleccionar conjunto de features a usar, ya sea descriptor, Onehot, NLP, FFT o propiedades fisicoquimicas. Las opcione son las siguiente.

Features: [seq_to_seq]

NLP: [bepler, fasttext, glove]

Onehot: [onehot]

FFT: [fft_alpha_structure, fft_betha_structure, fft_energetic, fft_hydropathy, fft_hydrophobicity, fft_index, fft_secondary_structure, fft_volume]

Physicochemical properties: [physicochemical_alpha_structure, physicochemical_betha_structure, physicochemical_energetic, physicochemical_hydropathy, physicochemical_hydrophobicity, physicochemical_index, physicochemical_secondary_structure, physicochemical_volume]

- **transform (opcional)**: seleccionar metodo de transformacion de features, ya sea PCA o Kernel-PCA [pca, kernel_pca]
 
- **epochs**: numero de epocas al momento de entrenar el modelo (GNN y MLP)

- **estimators**: numero de estimadores al entrenar el modelo de Random Forest
 
- **test**: seleccionar proporcion del set de datos que se usara para testear el modelo. [0, 1]

# Ejemplos

Modelos de regresion
```bash
python gnn_edge_regressor_pip.py --features bepler --transform pca --epochs 200 --test 0.3
python mlp_edge_regressor_pip.py --features bepler --transform kernel_pca --epochs 200 --test 0.3
python random_forest_edge_regressor_pip.py --features bepler --estimators 20 --test 0.3
```

Modelos de clasificacion
```bash
python gnn_linkpred_pip.py --features bepler --transform kernel_pca --epochs 200 --test 0.3
python mlp_linkpred_pip.py --features bepler  --epochs 200 --test 0.3
python random_forest_edge_regressor_pip.py --features bepler --transform pca --test 0.3
```


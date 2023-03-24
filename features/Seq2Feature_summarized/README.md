
### /biopython

##### Tareas asociadas :
1. Descarga o trabaja con proteínas conocidas, 1LM8, 1AQ2, proteínas relacionadas con P450, etc y comienza a aplicar grafos por distancia, las cuales sean del tipo euclideana o cosenos, o similares y que estén enfocadas en los centroides, carbonos alfa o átomos. Por favor, no emplees distancias del tipo relaciones. (Tarea 1)

2. Cuando calcules distancias, genera las distribuciones por histogramas y crea un test estadístico para determinar con cuales te quedarás, en ese sentido, normalmente nos qudaríamos con las primeras 25% (el primer cuartil) (Tarea 2)

* Comentarios:
    1. Falta hacerlo tipo pipeline, se cumple las tareas.

### /neo4j  
##### Tareas asociadas :

1. Construye los grafos y desarrolla los primeros análisis de topologías y descripciones del grafo en términos de nodos, aristas, grados, centralidades, etc. (Tarea 3)

* Comentarios:
    1. Esta carpeta contiene un script de parseo con output para ser cargado a Neo4j.
 
### /networkx
##### Tareas asociadas :

1. Construye los grafos y desarrolla los primeros análisis de topologías y descripciones del grafo en términos de nodos, aristas, grados, centralidades, etc. (Tarea 3)

* Comentarios:
    1. Getting started de la librería networkx para un ejemplo de grafo de proteinas.

### /Seq2Features_summarized

##### Tareas asociadas :

1. Explora las herramientas seq2feature para caracterizar secuencias y acóplalo a la librería de grafos que están armando. (Tarea 4)

* Comentarios:
    1. Se modifica el script para que pueda leer un archivo con N secuencias fastas.
    2. run: python3 sequence_based_features2.py --input_file fasta2.txt --property AAP --o example
    3. Fuente original del script: https://www.iitm.ac.in/bioinfo/SBFE/help.html




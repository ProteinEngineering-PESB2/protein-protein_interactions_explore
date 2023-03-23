import time
import sys
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from cnn_architectures import Models


df = pd.read_csv(sys.argv[1])
number_epochs = int(sys.argv[2])
architecture_option = sys.argv[3]
name_export = sys.argv[4]

response = df['class']
df_data = df.drop(columns=['class'])

X_train, X_test, y_train, y_test = train_test_split(df_data.values, response, test_size=0.3, random_state=42)

time_inicio = time.time()
models = Models(X_train, y_train, X_test, y_test, list(set(response)), architecture_option)
models.fit_models(number_epochs, 1)
metrics = models.get_metrics()
time_fin = time.time()
delta_time = round(time_fin - time_inicio, 4)

metrics["total_time"] = delta_time
metrics["epochs"] = number_epochs

with open(name_export, mode = "w", encoding = "utf-8") as file:
    json.dump(metrics, file)

    print(pd.json_normalize(metrics))
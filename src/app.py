from utils import db_connect
engine = db_connect()

# your code here

import requests
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split

# descargar data

url = "https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv"
nombre_archivo = "AB_NYC_2019.csv"

respuesta = requests.get(url)

with open(nombre_archivo, 'wb') as archivo:
    archivo.write(respuesta.content)

# convertir csv en dataframe

total_data = pd.read_csv("../data/raw/AB_NYC_2019.csv")
total_data.shape
total_data.info()  # 10 variables numericas # 6 variables categoricas # dos variables con gran numero de NaaN

# buscar duplicados

total_data_sin = total_data.drop_duplicates()
total_data_sin.shape

# Eliminar información irrelevante aunque no sabemos que problema se nos ha planteado. 
# Lo mas logico seria eliminar las siguentes columnas: id, name, host_id, host_name, calculated_host_listings_count
# tambien las columnas con alto numero de NaN: last_review, reviews_per_month 

total_data.drop(['id', 'host_id', 'name', 'host_name', 'last_review', 'reviews_per_month', 'calculated_host_listings_count'], axis = 1, inplace = True)
total_data.shape
print(total_data.head())

# draw histograms

fig, axis = plt.subplots(1, 3, figsize = (15, 7))
sns.histplot(ax = axis[0], data = total_data, x="neighbourhood_group").set(ylabel = None)
sns.histplot(ax = axis[1], data = total_data, x="neighbourhood").set(ylabel = None)
sns.histplot(ax = axis[2], data = total_data, x="room_type").set(ylabel = None)
plt.tight_layout()
plt.show()

# draw axis para analisis sobre variables numéricas

fig, axis = plt.subplots(2, 6, figsize = (20, 7))
sns.histplot(ax = axis[0, 0], data = total_data, x = "latitude")
sns.boxplot(ax = axis[1, 0], data = total_data, x = "latitude")
sns.histplot(ax = axis[0, 1], data = total_data, x = "longitude")
sns.boxplot(ax = axis[1, 1], data = total_data, x = "longitude")
sns.histplot(ax = axis[0, 2], data = total_data, x = "price")
sns.boxplot(ax = axis[1, 2], data = total_data, x = "price")
sns.histplot(ax = axis[0, 3], data = total_data, x = "minimum_nights")
sns.boxplot(ax = axis[1, 3], data = total_data, x = "minimum_nights")
sns.histplot(ax = axis[0, 4], data = total_data, x = "number_of_reviews")
sns.boxplot(ax = axis[1, 4], data = total_data, x = "number_of_reviews")
sns.histplot(ax = axis[0, 5], data = total_data, x = "availability_365")
sns.boxplot(ax = axis[1, 5], data = total_data, x = "availability_365")
plt.show()

# Dividir el conjunto en train y test 

X = total_data[['neighbourhood_group', 'neighbourhood', 'latitude', 'longitude', 'room_type', 'minimum_nights', 'number_of_reviews',
'availability_365']]  # Características (features)
y = total_data['price']  # Etiqueta (label)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
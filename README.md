# corte

import warnings
import pandas as pd
import numpy  as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


datos = pd.read_csv('../../../datos/MuestraCredito5000V2.csv', delimiter = ';', decimal = ".", header = 0)

# Convierte las variables de object a categórica
datos['IngresoNeto'] = datos['IngresoNeto'].astype('category')
datos['CoefCreditoAvaluo'] = datos['CoefCreditoAvaluo'].astype('category')
datos['MontoCuota'] = datos['MontoCuota'].astype('category')
datos['GradoAcademico'] = datos['GradoAcademico'].astype('category')

datos.info()
X = datos.loc[:, datos.columns != 'BuenPagador']
X

y = datos.loc[:, 'BuenPagador'].to_numpy()
y[0:6]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.85)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

preprocesamiento = ColumnTransformer(
  transformers=[
    ('cat', OneHotEncoder(sparse_output = False), ['IngresoNeto', 'CoefCreditoAvaluo', 'MontoCuota', 'GradoAcademico']),
    ('num', StandardScaler(), ['MontoCredito'])
  ]
)

#Utilizamos la función roc_curve de sklearn.metrics para obtener las proporciones de los #Verdaderos Positivos(Sensibilidad) y Falsos Positivos(Especificidad)

#Utilizamos la función roc_auc_score para obtener el valor del área bajo la curva.

#Verdaderos valores
Clase1 = np.array([0, 1, 0, 1, 1, 1, 0, 1])
# Las probabilidades de pertenecer a la clase positiva
Score1 = np.array([0.5, 0.4, 0.3, 0.9, 0.8, 0.5, 0.1, 0.9])

fpr1, tpr1, thresholds1 = roc_curve(Clase1, Score1, pos_label = 1)
print("False Positive Rate: ", fpr1)

print("True Positive Rate: ", tpr1)

print("Thresholds(Probabilidades de Corte): ", thresholds1)

area1 = roc_auc_score(Clase1, Score1)
print("Area: ", area1)
fig, ax = plt.subplots(1,1,figsize = (8,5), dpi = 150)
ax.plot([0,0.5,1],[0,0.5,1],color='black', linestyle = "dashed")
ax.plot(fpr1,tpr1,color='tab:blue', marker='o', label = "Ejemplo 1 (AUC: {})".format(area1))
no_print = ax.set_xlabel("Tasa de Falsos Positivos")
no_print = ax.set_ylabel("Tasa de Verdaderos Positivos")
no_print = ax.set_title("Curva ROC")
ax.legend(loc = "lower right")

#Verdaderos valores
Clase1 = np.array([0, 1, 0, 1, 1, 1, 0, 1])
# Las probabilidades de pertenecer a la clase positiva
Score2 = np.array([0.7, 0.8, 0.1, 0.4, 0.8, 0.2, 0.5, 0.3])
fpr2, tpr2, thresholds2 = roc_curve(Clase1, Score2, pos_label = 1)
print("False Positive Rate: ", fpr2)

print("True Positive Rate: ", tpr2)

print("Thresholds(Probabilidades de Corte): ", thresholds2)
area2 = roc_auc_score(Clase1, Score2)
print("Area: ", area2)
fig, ax = plt.subplots(1,1,figsize = (8,5), dpi = 150)
ax.plot([0,0.5,1], [0,0.5,1], color='black', linestyle = "dashed")
ax.plot(fpr1, tpr1, color='tab:blue', marker='o', label = "Ejemplo 1 (AUC: {})".format(area1))
ax.plot(fpr2, tpr2, color='tab:orange', marker='o', label = "Ejemplo 2 (AUC: {})".format(round(area2,2)))
no_print = ax.set_xlabel("Tasa de Falsos Positivos")
no_print = ax.set_ylabel("Tasa de Verdaderos Positivos")
no_print = ax.set_title("Curva ROC")
ax.legend(loc = "lower right")

#PROGRAMA 1: Códigos en Python para replicar los cálculos a mano.

def programa1(clase, score, labels, umbrales):
  # Graficamos la curva ROC
  fp_r, tp_r, umbral = roc_curve(clase, score, pos_label = "p")
  fig, ax = plt.subplots()
  no_print = ax.plot([0, 1], [0, 1], color = "black", linestyle = 'dashed')
  no_print = ax.plot(fp_r, tp_r)
  no_print = ax.set_xlabel("Tasa de Falsos Positivos")
  no_print = ax.set_ylabel("Tasa de Verdaderos Positivos")
  no_print = ax.set_title("Curva ROC")
  
  # Graficamos puntos con el siguiente algoritmo
  i = 1  # Contador
  
  for u in umbrales:
    Prediccion = np.where(score >= u, labels[0], labels[1])
    MC = confusion_matrix(clase, Prediccion)
    
    FP_r = round(MC[0, 1] / sum(MC[0, ]), 2)
    TP_r = round(MC[1, 1] / sum(MC[1, ]), 2)
    
    no_print = ax.plot(FP_r, TP_r, "o", color = "red")
    no_print = ax.annotate("T = " + str(round(u, 2)), (FP_r, TP_r + 0.02))
    
    # Imprimimos resultado
    print("=====================")
    print("Matriz ", i, "\n")  
    print("Probabilidad de Corte = T = ", round(u, 2), "\n")
    print("MC =")
    print(MC, "\n")
    print("Tasa FP = ", round(FP_r, 2), "\n")
    print("Tasa TP = ", round(TP_r, 2))     
    i = i + 1

clase = np.array([ "p",  "n",  "n",  "p",  "n",  "n",  "p",  "n",  "n",  "p"])
score = np.array([0.61, 0.06, 0.80, 0.11, 0.66, 0.46, 0.40, 0.19, 0.00, 0.91])
programa1(clase, score, ["p", "n"], np.arange(0, 1, 0.1))

#curva ROC con scikitlearn

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Verdaderos valores
Clase1 = np.array([ "p",  "n",  "n",  "p",  "n",  "n",  "p",  "n",  "n",  "p"])

# Las probabilidades de pertenecer a la clase positiva
Score1 = np.array([0.61, 0.06, 0.80, 0.11, 0.66, 0.46, 0.40, 0.19, 0.00, 0.91])

area1 = roc_auc_score(Clase1, Score1)
print("Area: ", area1)
fp_r, tp_r, umbral = roc_curve(Clase1, Score1, pos_label = "p")

fig, ax = plt.subplots()
no_print = ax.plot([0, 1], [0, 1], color = "black", linestyle = 'dashed')
no_print = ax.plot(fp_r, tp_r, label = "ROC 1: " + str(area1))
no_print = ax.set_xlabel("Tasa de Falsos Positivos")
no_print = ax.set_ylabel("Tasa de Verdaderos Positivos")
no_print = ax.set_title("Curva ROC")
no_print = ax.legend(loc = "lower right")
plt.show()

#curva roc con datos
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# SVM
# Para SVM es importante establecer el valor de TRUE en el parámetro de probability
# en caso de que se desee utilizar el método predict_proba()
svm_model = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', SVC(kernel = "rbf", probability = True))
])
svm_model.fit(X_train, y_train)

scores_svm = svm_model.predict_proba(X_test)
print(svm_model.classes_)

print(scores_svm[0:5,:])
# DECISION TREES
dt_model = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', DecisionTreeClassifier(criterion = "gini"))
])
dt_model.fit(X_train, y_train)
scores_dt = dt_model.predict_proba(X_test)
print(dt_model.classes_)
print(scores_dt[0:5,:])
# K NEAREST NEIGHBORS
knn_model = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', KNeighborsClassifier(n_neighbors = 6))
])
knn_model.fit(X_train, y_train)
scores_knn = knn_model.predict_proba(X_test)
print(knn_model.classes_)
print(scores_knn[0:5,:])

# RANDOM FOREST
rf_model = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestClassifier(n_estimators = 100, criterion = "gini"))
])
rf_model.fit(X_train, y_train)
scores_rf = rf_model.predict_proba(X_test)
print(rf_model.classes_)
print(scores_rf[0:5,:])
area_svm = roc_auc_score(y_test, scores_svm[:, 1])
area_svm

area_dt = roc_auc_score(y_test, scores_dt[:, 1])
area_dt
area_knn = roc_auc_score(y_test, scores_knn[:, 1])
area_knn

area_rf = roc_auc_score(y_test, scores_rf[:, 1])
area_rf

#Gráficar curva ROC
fp_svm, tp_svm, umbral_svm = roc_curve(y_test, scores_svm[:, 1], pos_label = "Si")
fp_dt, tp_dt, umbral_dt = roc_curve(y_test, scores_dt[:, 1], pos_label = "Si")
fp_knn, tp_knn, umbral_knn = roc_curve(y_test, scores_knn[:, 1], pos_label = "Si")
fp_rf, tp_rf, umbral_rf = roc_curve(y_test, scores_rf[:, 1], pos_label = "Si")

fig, ax = plt.subplots()
no_print = ax.plot([0, 1], [0, 1], color = "black", linestyle = 'dashed')

no_print = ax.plot(fp_svm, tp_svm, label = "SVM: " + str(round(area_svm, 3)))
no_print = ax.plot(fp_dt,  tp_dt,  label = "ARB: " + str(round(area_dt, 3)))
no_print = ax.plot(fp_knn, tp_knn, label = "KNN: " + str(round(area_knn, 3)))
no_print = ax.plot(fp_rf,  tp_rf,  label = "RNF: " + str(round(area_rf, 3)))

no_print = ax.set_xlabel("Tasa de Falsos Positivos")
no_print = ax.set_ylabel("Tasa de Verdaderos Positivos")
no_print = ax.set_title("Curva ROC")
no_print = ax.legend(loc = "lower right")
plt.show()


#Probabilidad de corte
#Este ajuste busca favorecer la predicción de más casos de la clase minoritaria, minimizando, en #la medida de lo posible, el desmejoramiento en la predicción de la clase mayoritaria.
#Generar modelo
instancia_rndf = Pipeline(steps=[
    ('preprocesador', preprocesamiento),
    ('clasificador', RandomForestClassifier(n_estimators = 100, criterion = "gini"))
])
instancia_rndf.fit(X_train, y_train)


#Obtener probabilidades
probabilidades = instancia_rndf.predict_proba(X_test)
probabilidades[0:5, :]
#Obtener probabilidades de una de las clases
probabilidad_si = probabilidades[:, 1]
probabilidad_si
#Generar indices con diferentes cortes
def indices_general(MC, nombres = None):
    precision_global = np.sum(MC.diagonal()) / np.sum(MC)
    error_global     = 1 - precision_global
    precision_categoria  = pd.DataFrame(MC.diagonal()/np.sum(MC,axis = 1)).T
    if nombres!=None:
        precision_categoria.columns = nombres
    return {"Matriz de Confusión":MC, 
            "Precisión Global":   precision_global, 
            "Error Global":       error_global, 
            "Precisión por categoría":precision_categoria}

# Aplicamos una Regla de Decisión
corte = np.arange(0, 1, 0.1)
tabla = []
for c in corte:
    print("===========================")
    print("Probabilidad de Corte: ",c)
    prediccion = np.where(probabilidad_si > c, "Si", "No") #where hace un arreglo con asignaciones dependiendo de la condición.
    # Calidad de la predicción 
    MC = confusion_matrix(y_test, prediccion)
    indices = indices_general(MC,list(np.unique(y)))
    ##### Tabla #####
    tabla = np.append(tabla,[ c, 1 - indices['Error Global']]) 
    precision_categoria = MC.diagonal()/np.sum(MC, axis=1)
    tabla = np.append(tabla,precision_categoria)
    ##### #####
    for k in indices:
        print("\n%s:\n%s"%(k,str(indices[k])))
tabla1 = np.array(tabla).reshape(10,4)
columnas =["Corte", "PG", "PN", "PP"]
df = pd.DataFrame(tabla1, columns = columnas)
df.Corte = ["%.2f" % x for x in df.Corte] #Se pasa el corte a un string
df














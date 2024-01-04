import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns,set()
%matplotlib inline

## Cargamos La base de Datos y la examinamos
Datos= pd.read_csv('C:\\Users\\livig\\OneDrive\\Documents\\Rstudio\\datos\\creditcard.csv')
Datos.head()
Datos.info()
Datos.describe()

## vamos a identificar cuantas son fraude y cuantas no
tipos = {0:'No fraude', 1:'Fraude'}
print(Datos.Class.value_counts().rename(index = tipos))

fig = plt.figure(figsize = (30, 27))

plt.subplot(10,11, 1) ; plt.plot(Datos.V1) ; plt.subplot(5, 6, 15) ; plt.plot(Datos.V15)
plt.subplot(5, 6, 2) ; plt.plot(Datos.V2) ; plt.subplot(5, 6, 16) ; plt.plot(Datos.V16)
plt.subplot(5, 6, 3) ; plt.plot(Datos.V3) ; plt.subplot(5, 6, 17) ; plt.plot(Datos.V17)
plt.subplot(5, 6, 4) ; plt.plot(Datos.V4) ; plt.subplot(5, 6, 18) ; plt.plot(Datos.V18)
plt.subplot(5, 6, 5) ; plt.plot(Datos.V5) ; plt.subplot(5, 6, 19) ; plt.plot(Datos.V19)
plt.subplot(5, 6, 6) ; plt.plot(Datos.V6) ; plt.subplot(5, 6, 20) ; plt.plot(Datos.V20)
plt.subplot(5, 6, 7) ; plt.plot(Datos.V7) ; plt.subplot(5, 6, 21) ; plt.plot(Datos.V21)
plt.subplot(5, 6, 8) ; plt.plot(Datos.V8) ; plt.subplot(5, 6, 22) ; plt.plot(Datos.V22)
plt.subplot(5, 6, 9) ; plt.plot(Datos.V9) ; plt.subplot(5, 6, 23) ; plt.plot(Datos.V23)
plt.subplot(5, 6, 10) ; plt.plot(Datos.V10) ; plt.subplot(5, 6, 24) ; plt.plot(Datos.V24)
plt.subplot(5, 6, 11) ; plt.plot(Datos.V11) ; plt.subplot(5, 6, 25) ; plt.plot(Datos.V25)
plt.subplot(5, 6, 12) ; plt.plot(Datos.V12) ; plt.subplot(5, 6, 26) ; plt.plot(Datos.V26)
plt.subplot(5, 6, 13) ; plt.plot(Datos.V13) ; plt.subplot(5, 6, 27) ; plt.plot(Datos.V27)
plt.subplot(5, 6, 14) ; plt.plot(Datos.V14) ; plt.subplot(5, 6, 28) ; plt.plot(Datos.V28)
plt.subplot(5, 6, 29) ; plt.plot(Datos.Amount)
plt.show()

##from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

feature_names = Datos.iloc[:,1:30].columns
target = Datos.iloc[:1,30:].columns
print(feature_names)
print(target)

data_features = Datos[feature_names]
data_target = Datos[target]

X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=1)
print("Length of X_train is: {X_train}".format(X_train = len(X_train)))
print("Length of X_test is: {X_test}".format(X_test = len(X_test)))
print("Length of y_train is: {y_train}".format(y_train = len(y_train)))
print("Length of y_test is: {y_test}".format(y_test = len(y_test)))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train.values.ravel())

pred = model.predict(X_test)

class_names = ['not_fraud', 'fraud']
matrix = confusion_matrix(y_test, pred)
# Create pandas dataframe
dataframe = pd.DataFrame(matrix, index=class_names, columns=class_names)
# Create heatmap
sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt = 'g')
plt.title("Confusion Matrix"), plt.tight_layout()
plt.ylabel("True Class"), plt.xlabel("Predicted Class")
plt.show()

from sklearn.metrics import f1_score, recall_score
f1_score = round(f1_score(y_test, pred), 2)
recall_score = round(recall_score(y_test, pred), 2)
print("Sensitivity/Recall for Logistic Regression Model 1 : {recall_score}".format(recall_score = recall_score))
print("F1 Score for Logistic Regression Model 1 : {f1_score}".format(f1_score = f1_score))

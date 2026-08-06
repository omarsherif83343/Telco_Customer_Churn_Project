import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

#Stage 1

df = pd.read_csv('C:\\telco-customer-churn.csv')

print(df.head())
print(df.info())
print(df.describe())
print("-------------------------------------------------------------")
print(df["customerID"].count())
print(len(df.columns))
print(df.isnull().sum())
print(len(df[df["Churn"] == "Yes"]))
print(len(df[df["Churn"] == "No"]))
print("---------------------------------------------------------------------------------")
df["TotalCharges"] = df["TotalCharges"].replace(" ",np.nan)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")
df.dropna(inplace=True)
#---------------------------------------------------------------------------------------

#Stage 2
print(len(df[df.duplicated() == True]))

#----------------------------------------------------------------------------------------

#Stage 3

del df["customerID"]
print(df.dtypes)

encoder = LabelEncoder()
df["gender"] = encoder.fit_transform(df["gender"])
df["Partner"] = encoder.fit_transform(df["Partner"])
df["Dependents"] = encoder.fit_transform(df["Dependents"])
df["PhoneService"] = encoder.fit_transform(df["PhoneService"])
df["PaperlessBilling"] = encoder.fit_transform(df["PaperlessBilling"])
df["Churn"] = encoder.fit_transform(df["Churn"])

df = pd.get_dummies(df,columns=["MultipleLines"])
df = pd.get_dummies(df,columns=["InternetService"])
df = pd.get_dummies(df,columns=["Contract"])
df = pd.get_dummies(df,columns=["PaymentMethod"])
df = pd.get_dummies(df,columns=["OnlineSecurity"])
df = pd.get_dummies(df,columns=["OnlineBackup"])
df = pd.get_dummies(df,columns=["DeviceProtection"])
df = pd.get_dummies(df,columns=["TechSupport"])
df = pd.get_dummies(df,columns=["StreamingTV"])
df = pd.get_dummies(df,columns=["StreamingMovies"])

#---------------------------------------------------------------------

#Stage 4

X = df.drop("Churn",axis=1)
y = df["Churn"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

accuracies = []
k_values = range(1,22,2)

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train,y_train)
    knn_pred = knn_model.predict(X_test)
    accuracies.append(accuracy_score(y_test,knn_pred))

best_k = k_values[np.argmax(accuracies)]
best_accuracies = max(accuracies)

print("Best K:",best_k)
print("Best Accuracies:",best_accuracies)

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train,y_train)
final_knn_pred = final_knn.predict(X_test)
matrix = confusion_matrix(y_test,final_knn_pred)
report = classification_report(y_test,final_knn_pred)



#---------------------------------------------------------------------------
#Stage 5

plt.plot(k_values,accuracies,marker='o')
plt.xlabel("Number Of Neighbors (K)")
plt.ylabel("Accuracy")
plt.title("K Vs Accuracy")
plt.grid(True)
plt.savefig("knn_accuracy.png")
plt.show()


sns.heatmap(matrix,annot=True,fmt="d",cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("\nConfusion Matrix:")
print(matrix)
print("Classification Report\n")
print(report)
print("Dataset Shape:",df.shape)
print("Training Samples:",X_train.shape[0])
print("Testing Samples:",X_test.shape[0])
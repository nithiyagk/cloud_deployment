# train.py
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

df = pd.read_csv("logistic.csv")

df.isnull().sum()
df.duplicated().sum()
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

df = df.drop("User ID",axis = 1)
X = df.drop("Purchased" ,axis = 1)
y = df["Purchased"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.fit_transform(X_test)

model = LogisticRegression()
model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy_score(y_test,pred)

import joblib
joblib.dump(model, "logistic regression.pkl")
# ==== 0) Imports ====
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==== 1) Extracción de datos (Titanic - Seaborn) ====
def loadDataset():
    df = sns.load_dataset("titanic")
    # Quitamos columnas con demasiados nulos para simplificar
    df = df.drop(columns=["deck", "embark_town", "alive", "class", "who", "adult_male", "alone"])
    return df

df = loadDataset()

print(" ==== 1) Extracción de datos (Titanic - Seaborn) ====")
print(df)

# # ==== 2) Filtrado básico ====
def filterDataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    # Ejemplo: mantener solo edades válidas
    if "age" in df:
        df = df[(df["age"].isna()) | ((df["age"] >= 0) & (df["age"] <= 100))]
    return df

print(" ==== 2) Filtrado básico ====")
df = filterDataframe(df)
print(df)

# # ==== 3) Separación de variables y target ====
targetCol = "survived"
X = df.drop(columns=[targetCol])
y = df[targetCol]

# Detectar tipos automáticamente
numericFeatures = selector(dtype_include=np.number)(X)
categoricalFeatures = selector(dtype_include=["object", "category"])(X)

print(" ==== 3) Separación de variables y target ====")
print("Columnas numéricas:", numericFeatures)
print("Columnas categóricas:", categoricalFeatures)

# ==== 4) Preprocesamiento por tipo + modelo en un Pipeline ====
numericPipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categoricalPipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numericPipeline, numericFeatures),
        ("cat", categoricalPipeline, categoricalFeatures),
    ],
    remainder="drop"
)

model = LogisticRegression(max_iter=1000)

mlPipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", model)
])

print(" ==== 4) Preprocesamiento por tipo + modelo en un Pipeline ====")
print(model)

# # ==== 5) Split y entrenamiento ====

print(" ==== 5) Split y entrenamiento ====")
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

mlPipeline.fit(XTrain, yTrain)

# # ==== 6) Evaluación ====

print(" ==== 6) Evaluación ====")
yPred = mlPipeline.predict(XTest)
acc = accuracy_score(yTest, yPred)

print(f"Shape train: {XTrain.shape}, test: {XTest.shape}")
print(f"Accuracy en test: {acc:.3f}")

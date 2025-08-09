# ==== 0) Imports ====
import seaborn as sns
import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==== 1) Definición de columnas ====
set_config(display="diagram")

numericFeatures = ["age", "fare", "sibsp", "parch"]
categoricalFeatures = ["sex", "embarked", "pclass"]
targetCol = "survived"

# ==== 2) Cargar dataset (Titanic - seaborn) ====
df = sns.load_dataset("titanic")

# (Opcional) quitar columnas que no usaremos para que no estorben:
# df = df[numericFeatures + categoricalFeatures + [targetCol]]

# ==== 3) Separar features/target ====
X = df[numericFeatures + categoricalFeatures]
y = df[targetCol]

# ==== 4) Pipelines de preprocesamiento ====
numericPipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categoricalPipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # salida sparse por defecto
])

preprocess = ColumnTransformer([
    ("numerical", numericPipeline, numericFeatures),
    ("categorical", categoricalPipeline, categoricalFeatures),
])

# ==== 5) Pipeline final (tu forma) ====
mlPipeline = Pipeline([
    ("preprocessor", preprocess),
    ("svd", TruncatedSVD(n_components=10, random_state=0)),
    ("clf", LogisticRegression(C=0.1, max_iter=10000, random_state=0, solver="newton-cg")),
])

# ==== 6) Split, entrenamiento y evaluación ====
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

mlPipeline.fit(XTrain, yTrain)
yPred = mlPipeline.predict(XTest)
acc = accuracy_score(yTest, yPred)

# ==== 7) Resultados ====
print(f"Columnas numéricas: {numericFeatures}")
print(f"Columnas categóricas: {categoricalFeatures}")
print(f"Accuracy en test: {acc:.3f}")

mlPipeline

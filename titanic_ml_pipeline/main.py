import seaborn as sns
import joblib
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Opcional: muestra diagrama del pipeline si lo ejecutas en notebook
set_config(display="diagram")

# Columnas
numericFeatures = ["age", "fare", "sibsp", "parch"]
categoricalFeatures = ["sex", "embarked", "pclass"]
targetCol = "survived"

def main(modelPath: str = "titanic_pipeline.joblib"):
    # 1) Cargar dataset (Titanic - seaborn)
    df = sns.load_dataset("titanic")

    # 2) Seleccionar features/target
    X = df[numericFeatures + categoricalFeatures]
    y = df[targetCol]

    # 3) Preprocesamiento
    numericPipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categoricalPipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))  # salida sparse
    ])
    preprocess = ColumnTransformer([
        ("numerical", numericPipeline, numericFeatures),
        ("categorical", categoricalPipeline, categoricalFeatures),
    ])

    # 4) Pipeline final
    mlPipeline = Pipeline([
        ("preprocessor", preprocess),
        ("svd", TruncatedSVD(n_components=10, random_state=0)),
        ("clf", LogisticRegression(C=0.1, max_iter=10000, random_state=0, solver="newton-cg")),
    ])

    # 5) Split + fit + eval
    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mlPipeline.fit(XTrain, yTrain)
    yPred = mlPipeline.predict(XTest)
    acc = accuracy_score(yTest, yPred)

    print(f"Accuracy en test: {acc:.3f}")
    joblib.dump(mlPipeline, modelPath)
    print(f"Modelo guardado en {modelPath}")

if __name__ == "__main__":
    main()

import seaborn as sns
import joblib
import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

# Opcional: muestra diagrama del pipeline si lo ejecutas en notebook
set_config(display="diagram")

# Columnas
numericFeatures = ["age", "fare", "sibsp", "parch"]
categoricalFeatures = ["sex", "embarked", "pclass"]
targetCol = "survived"

def main(modelPath: str = "titanic_pipeline.joblib", resultsCsv: str | None = "grid_results.csv"):
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

    mlPipeline = Pipeline([
        ("preprocessor", preprocess),
        ("svd", TruncatedSVD(random_state=0)),
        ("clf", LogisticRegression(max_iter=10000, random_state=0))
    ])

    paramGrid = {
        "svd__n_components": [10, 20, 30, 50],
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["lbfgs", "newton-cg"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=mlPipeline,
        param_grid=paramGrid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    XTrain, XTest, yTrain, yTest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    grid.fit(XTrain, yTrain)

    print("Mejores hiperparámetros:", grid.best_params_)
    print(f"CV best_score (accuracy): {grid.best_score_:.3f}")

    results = pd.DataFrame(grid.cv_results_).sort_values(
        by="mean_test_score", ascending=False
    )
    print("\nTop combinaciones por mean_test_score:")
    print(results[["mean_test_score", "std_test_score", "mean_train_score", "params"]].head(15))

    bestModel = grid.best_estimator_
    yPred = bestModel.predict(XTest)
    acc = accuracy_score(yTest, yPred)
    print(f"Accuracy en test: {acc:.3f}")

    joblib.dump(bestModel, modelPath)
    print(f"Modelo (mejor estimador) guardado en {modelPath}")

if __name__ == "__main__":
    main()

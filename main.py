import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score

# Database Read
dataset_path = "data/data.csv"
df = pd.read_csv(dataset_path)

# Exploração inicial
print("Format: ", df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())
print(df["diagnosis"].value_counts())

# Cleaning
df = df.drop(columns=["Unnamed: 32"])
df = df.drop(columns=["id"])

# Diagnosis Encoding
df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})

correlation = df.corr()

# Graphic Correlation
correlation_diagnostics = (
    correlation["diagnosis"]
    .drop("diagnosis")
    .sort_values()
)

plt.figure(figsize=(10, 8))

correlation_diagnostics.plot(kind="barh")

plt.title("Correlação das Features com o Diagnóstico")
plt.xlabel("Correlação")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("correlacao_diagnostico.png")
plt.close()

# Distribution of radius_mean
plt.figure(figsize=(8, 6))

sns.histplot(
    data=df,
    x="radius_mean",
    hue="diagnosis",
    kde=True
)

plt.title("Distribuição de Radius Mean por Diagnóstico")
plt.xlabel("Radius Mean")
plt.ylabel("Quantidade")

plt.tight_layout()
plt.savefig("radius_mean_diagnostico.png")
plt.close()

# Concave_points_mean_diagnostico
plt.figure(figsize=(8, 6))

sns.histplot(
    data=df,
    x="concave points_mean",
    hue="diagnosis",
    kde=True
)

plt.title("Distribuição de Concave Points Mean por Diagnóstico")
plt.xlabel("Concave Points Mean")
plt.ylabel("Quantidade")

plt.tight_layout()
plt.savefig("concave_points_mean_diagnostico.png")
plt.close()

# Bloxplt radius
plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="diagnosis",
    y="radius_mean"
)

plt.title("Distribuição de Radius Mean por Diagnóstico")
plt.xlabel("Diagnóstico")
plt.ylabel("Radius Mean")

plt.tight_layout()
plt.savefig("boxplot_radius_mean.png")
plt.close()

# X = as características do exame, y = o diagnóstico que queremos prever
X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

# Separação: 60% treino, 20% validação, 20% teste
# stratify=y mantém a mesma proporção de benignos e malignos em cada parte
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42
)

# Pipeline: o pré-processamento e o modelo ficam juntos
modelos = {
    "Regressão Logística": Pipeline([
        ("scaler", StandardScaler()),   # coloca todas as colunas na mesma escala
        ("modelo", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("modelo", RandomForestClassifier(n_estimators=200, random_state=42))
    ]),
}

# Treina com o treino e compara os modelos na validação
for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    recall = recall_score(y_val, modelo.predict(X_val))
    print(f"{nome} - Recall na validação: {recall:.3f}")


resultados = {}

for nome, modelo in modelos.items():
    modelo.fit(X_train, y_train)

    y_val_pred = modelo.predict(X_val)
    recall = recall_score(y_val, y_val_pred)

    resultados[nome] = recall

    print(f"{nome} - Recall na validação: {recall:.3f}")

melhor_nome = max(resultados, key=resultados.get)
better = modelos[melhor_nome]

print(f"\nModelo selecionado: {melhor_nome}")
print(f"Recall na validação: {resultados[melhor_nome]:.3f}")

# Avaliação final: utilizar o conjunto de teste somente agora
y_pred = better.predict(X_test)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

y_pred = better.predict(X_test)


# Feature Importance - Random Forest

rf_model = modelos["Random Forest"]

importancias = pd.Series(
    rf_model.named_steps["modelo"].feature_importances_,
    index=X_train.columns,
    name="Importância"
).sort_values(ascending=False)

# Exibe as 10 características mais importantes
top_10 = importancias.head(10)

print("\nTop 10 características mais importantes:")
print(top_10.to_string())

# Gera o gráfico
plt.figure(figsize=(10, 6))

top_10.sort_values().plot(kind="barh")

plt.title("Top 10 - Feature Importance (Random Forest)")
plt.xlabel("Importância")
plt.ylabel("Característica")

plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.close()

print("\nGráfico salvo em feature_importance.png")

print(f"Modelo selecionado: {melhor_nome}")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
print(
    "Precisão (maligno):",
    f"{precision_score(y_test, y_pred, pos_label=1):.4f}"
)
print(
    "Recall (maligno):",
    f"{recall_score(y_test, y_pred, pos_label=1):.4f}"
)
print(
    "F1-score (maligno):",
    f"{f1_score(y_test, y_pred, pos_label=1):.4f}"
)

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\nRelatório completo:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Benigno", "Maligno"],
    digits=4,
))

print("\nResultados no conjunto de teste:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Benigno", "Maligno"],
    zero_division=0
))
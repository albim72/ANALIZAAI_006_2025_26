import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(layout="wide")

# ---- ŁADOWANIE DANYCH ----
st.title("Boston Housing: AI & Data Science Dashboard")

boston = fetch_openml(name='boston', version=1, as_frame=True)
data = boston.frame

st.write("Próbka danych (Boston Housing):")
st.dataframe(data.head())

# ---- EKSPLORACJA ----
st.header("Eksploracja danych")
feature = st.selectbox("Wybierz cechę do analizy:", data.columns[:-1])
fig, ax = plt.subplots()
ax.hist(data[feature], bins=30, alpha=0.7)
ax.set_title(f"Rozkład cechy: {feature}")
st.pyplot(fig)

# ---- MODELOWANIE AI ----
st.header("Model AI: Predykcja cen mieszkań")
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Wyniki
st.subheader("Wyniki modelu (RandomForest)")
st.write(f"Średni błąd absolutny: {mean_absolute_error(y_test, y_pred):.2f}")
st.write(f"R²: {r2_score(y_test, y_pred):.2f}")

# ---- WYKRES: Prawdziwe vs Predykowane ----
fig2, ax2 = plt.subplots()
ax2.scatter(y_test, y_pred, alpha=0.6)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax2.set_xlabel("Wartość rzeczywista (MEDV)")
ax2.set_ylabel("Predykcja modelu")
ax2.set_title("Predykcja vs Rzeczywistość")
st.pyplot(fig2)

# ---- WIZUALIZACJA WAŻNOŚCI CECH ----
st.subheader("Najważniejsze cechy wg AI")
importances = model.feature_importances_
importance_df = pd.DataFrame({"cecha": X.columns, "ważność": importances}).sort_values("ważność", ascending=False)
st.bar_chart(importance_df.set_index("cecha"))

# ---- PREDYKTOR INTERAKTYWNY ----
st.header("Interaktywny predyktor ceny mieszkania")
user_input = {}
for col in X.columns:
    user_input[col] = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
user_df = pd.DataFrame([user_input])
prediction = model.predict(user_df)[0]
st.success(f"Przewidywana cena mieszkania: **{prediction:.2f}** (MEDV)")

st.caption("Źródło danych: openml.org, model: Random Forest, dashboard: Streamlit, autor: ScikitLearn")

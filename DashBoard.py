import streamlit as st
import pandas as pd
import plotly.express as px


# configurando o layout
st.set_page_config(layout="wide")

# Carregando o novo dataset com codificação especificada
df = pd.read_csv("car_ad.csv", sep=";", decimal=",", encoding="latin1")
df["Data"] = pd.to_datetime(df["Data"])
df = df.sort_values("Data")

# Criando a coluna de mês
df["Mês"] = df["Data"].apply(lambda x: str(x.year) + "-" + str(x.month))
month = st.sidebar.selectbox("Mês", df["Mês"].unique())
df_filtered = df[df["Mês"] == month]

# Filtros na barra lateral
marcas = st.sidebar.multiselect("Marca", df["Marca"].unique(), default=df["Marca"].unique())
st.sidebar.image("fatec_pompeia.jpg", use_container_width=True)

if marcas:
    df_filtered = df_filtered[df_filtered["Marca"].isin(marcas)]

# Título do dashboard
st.title("Dashboard de Análise de Carros")
st.write("Business Intelligence")

# Resumo dos dados
st.markdown("## Resumo")
total_anuncios = df_filtered.shape[0]
preco_medio = df_filtered["Preço"].mean()
km_medio = df_filtered["Quilometragem"].mean()
total_marcas = df_filtered["Marca"].nunique()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total de Anúncios", value=total_anuncios)

with col2:
    st.metric(label="Preço Médio", value=f"R${preco_medio:.2f}".replace(".", ","))

with col3:
    st.metric(label="Quilometragem Média", value=f"{km_medio:.2f} km")

with col4:
    st.metric(label="Total de Marcas", value=total_marcas)

# Gráficos
col1, col2 = st.columns(2)
col3, col4, col5 = st.columns(3)

fig_date = px.bar(df_filtered, x="Data", y="Preço", color="Marca", title="Preço por Data")
col1.plotly_chart(fig_date, use_container_width=True)

fig_km = px.bar(df_filtered, x="Data", y="Quilometragem", color="Marca", title="Quilometragem por Data", orientation="h")
col2.plotly_chart(fig_km, use_container_width=True)

marca_total = df_filtered.groupby("Marca")["Preço"].mean().reset_index()
fig_marca = px.bar(marca_total, x="Marca", y="Preço", title="Preço Médio por Marca")
col3.plotly_chart(fig_marca, use_container_width=True)

fig_tipo = px.pie(df_filtered, values="Preço", names="Tipo", title="Distribuição por Tipo de Veículo")
col4.plotly_chart(fig_tipo, use_container_width=True)

fig_km_marca = px.bar(df_filtered, x="Marca", y="Quilometragem", title="Quilometragem Média por Marca")
col5.plotly_chart(fig_km_marca, use_container_width=True)

# Recomendação de veículos
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.markdown("### Recomendação de Veículos")

X = df[["Marca", "Tipo", "Preço"]]
y = df["Modelo"]

le_marca = LabelEncoder()
le_tipo = LabelEncoder()
le_modelo = LabelEncoder()

X["Marca"] = le_marca.fit_transform(X["Marca"])
X["Tipo"] = le_tipo.fit_transform(X["Tipo"])
y_encoded = le_modelo.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.15, random_state=0)
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

st.markdown("### Insira os detalhes do veículo para receber uma recomendação")
marca_input = st.selectbox("Marca", df["Marca"].unique())
tipo_input = st.selectbox("Tipo", df["Tipo"].unique())
preco_input = st.number_input("Preço", min_value=0.0, step=0.01)

novo_veiculo = pd.DataFrame([[le_marca.transform([marca_input])[0], le_tipo.transform([tipo_input])[0], preco_input]], columns=["Marca", "Tipo", "Preço"])

novo_veiculo_scaled = scaler.transform(novo_veiculo)

predicao = knn.predict(novo_veiculo_scaled)

modelo_recomendado = le_modelo.inverse_transform(predicao)

st.markdown('### Modelo recomendado para o veículo')
st.success(f"Modelo recomendado: {modelo_recomendado[0]}")
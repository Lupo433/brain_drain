import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import Normalize
from matplotlib import cm

# === Carica il dataset ===
df = pd.read_csv("dataset_final.csv")  # assicurati che il file esista nel root

st.set_page_config(page_title="Brain Drain Analyzer", page_icon="🌍", layout="wide")

st.title("🌍 GoWhere - Brain Drain Analyzer")

# === INDICI DISPONIBILI ===
indici_disponibili = [
    "Education", "Jobs", "Income", "Safety", "Health", "Environment",
    "Civic engagement", "Accessiblity to services", "Housing",
    "Community", "Life satisfaction", "PR rating", "CL rating"
]

# === Selezione utente ===
st.subheader("1️⃣ Il tuo profilo")

sex = st.selectbox("Sesso", ["Male", "Female"])
origin = st.selectbox("Paese di origine", sorted(df["country_of_birth"].unique()))

st.subheader("2️⃣ Cosa vuoi migliorare nel tuo Paese?")
indici_da_migliorare = {}
for ind in indici_disponibili:
    peso = st.slider(f"{ind}", 0.0, 10.0, 3.0, 0.5, key=f"neg_{ind}")
    if peso > 0:
        indici_da_migliorare[ind] = peso

st.subheader("3️⃣ Cosa desideri trovare nel nuovo Paese?")
indici_desiderati = {}
for ind in indici_disponibili:
    peso = st.slider(f"{ind}", 0.0, 10.0, 3.0, 0.5, key=f"pos_{ind}")
    if peso > 0:
        indici_desiderati[ind] = peso

st.markdown("---")

# === Funzione raccomandazione ===
def consiglia_paesi(df, sex, origin, migliora, desidera, top_n=5):
    df_user = df[(df["country_of_birth"] == origin) & (df["sex"] == sex)].copy()

    def calcola_score(row):
        score = 0
        motivi = []

        for ind, peso in migliora.items():
            delta = row[f"dest_{ind}"] - row[f"origin_{ind}"]
            score += delta * peso
            if delta > 0:
                motivi.append(f"{ind} ↑ (+{delta:.2f})")
            elif delta < 0:
                motivi.append(f"{ind} ↓ ({delta:.2f})")

        for ind, peso in desidera.items():
            val = row[f"dest_{ind}"]
            delta = val - row[f"origin_{ind}"]
            score += val * peso
            if delta > 0:
                motivi.append(f"{ind} ↑ (+{delta:.2f})")
            elif delta < 0:
                motivi.append(f"{ind} ↓ ({delta:.2f})")
            else:
                motivi.append(f"{ind} = ({val:.2f})")

        return pd.Series({"score": score, "motivi": ", ".join(motivi)})

    df_user[["score", "motivi"]] = df_user.apply(calcola_score, axis=1)
    df_user["score_norm"] = MinMaxScaler().fit_transform(df_user[["score"]]) if df_user["score"].nunique() > 1 else 1.0

    if desidera:
        profile = np.array([peso for peso in desidera.values()]).reshape(1, -1)
        columns = [f"dest_{k}" for k in desidera.keys()]
        similarity = cosine_similarity(profile, df_user[columns])[0]
    else:
        similarity = np.zeros(len(df_user))

    df_user["similarity"] = similarity
    df_user["score_finale"] = 0.5 * df_user["score_norm"] + 0.5 * df_user["similarity"]

    ranking = (
        df_user.groupby("country_of_destination")
        .agg({
            "score_finale": "mean",
            "motivi": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
        })
        .sort_values("score_finale", ascending=False)
        .reset_index()
        .head(top_n)
    )

    return ranking

# === Bottone per generare risultati ===
if st.button("🔍 Scopri i paesi migliori"):
    user_input = {
        "sex": sex,
        "origin_country": origin,
        "indici_da_migliorare": indici_da_migliorare,
        "indici_desiderati": indici_desiderati
    }

    risultato = consiglia_paesi(df, **user_input)
    
    st.subheader("📊 Paesi consigliati:")
    st.dataframe(risultato)

    st.subheader("📈 Punteggi Normalizzati")
    fig, ax = plt.subplots()
    ax.barh(risultato["country_of_destination"], risultato["score_finale"], color="mediumseagreen")
    ax.set_xlabel("Score finale combinato")
    ax.set_title("Top Paesi Consigliati")
    ax.invert_yaxis()
    st.pyplot(fig)

# === Confronto Paesi ===
st.markdown("---")
st.subheader("📊 Confronta due Paesi")

col1, col2 = st.columns(2)
with col1:
    paese1 = st.selectbox("Paese 1", sorted(df["country_of_destination"].unique()))
with col2:
    paese2 = st.selectbox("Paese 2", sorted(df["country_of_destination"].unique()), index=1)

indici_confronto = st.multiselect(
    "Scegli gli indicatori per il confronto",
    options=indici_disponibili,
    default=["Jobs", "Income"]
)

if st.button("📊 Confronta Paesi"):
    p1 = df[df["country_of_destination"] == paese1][[f"dest_{i}" for i in indici_confronto]].mean()
    p2 = df[df["country_of_destination"] == paese2][[f"dest_{i}" for i in indici_confronto]].mean()

    p1.index = [i.replace("dest_", "") for i in p1.index]
    p2.index = [i.replace("dest_", "") for i in p2.index]

    x = np.arange(len(indici_confronto))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, p1.values, width=width, label=paese1)
    ax.bar(x + width / 2, p2.values, width=width, label=paese2)
    ax.set_xticks(x)
    ax.set_xticklabels(indici_confronto, rotation=45)
    ax.legend()
    st.pyplot(fig)

# === Diagramma Hasse ===
st.markdown("---")
st.subheader("📌 Visualizza Diagramma di Hasse")

with st.expander("Mostra diagramma Hasse personalizzato"):
    dest_cols = [col for col in df.columns if col.startswith("dest_")]

    var1 = st.selectbox("Variabile 1", dest_cols)
    var2 = st.selectbox("Variabile 2", dest_cols, index=1)
    var3 = st.selectbox("Variabile 3", dest_cols, index=2)
    color_metric = st.selectbox("Colore nodi per:", dest_cols, index=3)

    if st.button("📌 Mostra Hasse"):
        def dominates(a, b):
            return all(a >= b) and any(a > b)

        selected_vars = [var1, var2, var3]
        df_grouped = df.groupby("country_of_destination")[selected_vars + [color_metric]].mean()

        G = nx.DiGraph()
        for i in df_grouped.index:
            for j in df_grouped.index:
                if i == j:
                    continue
                a = df_grouped.loc[i, selected_vars]
                b = df_grouped.loc[j, selected_vars]
                if dominates(a, b):
                    intermedi = [
                        k for k in df_grouped.index
                        if dominates(df_grouped.loc[i, selected_vars], df_grouped.loc[k, selected_vars])
                        and dominates(df_grouped.loc[k, selected_vars], df_grouped.loc[j, selected_vars])
                        and k != i and k != j
                    ]
                    if not intermedi:
                        G.add_edge(i, j)

        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            pos = graphviz_layout(G, prog="dot")
        except:
            st.error("Errore: pygraphviz non installato")
            pos = nx.spring_layout(G)

        color_vals = df_grouped[color_metric].to_dict()
        node_colors = [color_vals.get(n, 0.5) for n in G.nodes()]
        cmap = cm.get_cmap("Blues")
        norm = Normalize(vmin=min(node_colors), vmax=max(node_colors))
        node_sizes = [800 + 400 * G.out_degree(n) for n in G.nodes()]

        fig, ax = plt.subplots(figsize=(14, 10))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                               cmap=cmap, ax=ax, edgecolors='black')
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True,
                               arrowstyle='-|>', arrowsize=20, edge_color='gray')
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=9)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label(color_metric)

        ax.set_title("Hasse Diagram")
        ax.axis("off")
        st.pyplot(fig)

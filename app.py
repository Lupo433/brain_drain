import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import Normalize
from matplotlib import cm

# === DATI ===
df = pd.read_csv("dataset_final.csv")
dest_cols = [col for col in df.columns if col.startswith("dest_")]
valid_dest_cols = [col for col in dest_cols if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().any()]

# === UTILS ===
def dominates(a, b):
    return all(a >= b) and any(a > b)

# === LAYOUT ===
st.set_page_config(page_title="GoWhere - Brain Drain Analyzer", layout="wide")
st.title("🌍 GoWhere - Brain Drain Analyzer")
st.markdown("""
Questo strumento ti aiuta a scoprire i paesi migliori per le tue esigenze personali, confrontando indicatori chiave come lavoro, salute, ambiente e altro. 
Rispondi alle domande, scopri raccomandazioni personalizzate e confronta visivamente le opzioni disponibili.
""")

# === USER INPUT ===
st.sidebar.header("📥 Input Utente")
sex = st.sidebar.selectbox("Sesso", ["Male", "Female"])
origin = st.sidebar.selectbox("Paese di origine", sorted(df["country_of_birth"].unique()))

indici = [
    "Education", "Jobs", "Income", "Safety", "Health", "Environment",
    "Civic engagement", "Accessiblity to services", "Housing",
    "Community", "Life satisfaction", "PR rating", "CL rating"
]

st.sidebar.markdown("### ❌ Cosa vuoi migliorare dal tuo paese attuale?")
migliora = {ind: st.sidebar.slider(f"{ind} ❌", 0.0, 10.0, 0.0, 0.5) for ind in indici}
migliora = {k: v for k, v in migliora.items() if v > 0}

st.sidebar.markdown("### ✅ Cosa desideri nel nuovo paese?")
desidera = {ind: st.sidebar.slider(f"{ind} ✅", 0.0, 10.0, 0.0, 0.5) for ind in indici}
desidera = {k: v for k, v in desidera.items() if v > 0}

user_input = {"sex": sex, "origin": origin, "migliora": migliora, "desidera": desidera}

# === RACCOMANDAZIONE ===
def consiglia_paesi(df, sex, origin, migliora, desidera, top_n=5):
    df_user = df[(df["country_of_birth"] == origin) & (df["sex"] == sex)].copy()

    def calcola_punteggio(r):
        score = 0
        motivi = []
        for ind, peso in migliora.items():
            delta = r[f"dest_{ind}"] - r[f"origin_{ind}"]
            score += delta * peso
            if delta > 0:
                motivi.append(f"{ind} ↑ (+{delta:.2f})")
            elif delta < 0:
                motivi.append(f"{ind} ↓ ({delta:.2f})")
        for ind, peso in desidera.items():
            val = r[f"dest_{ind}"]
            delta = r[f"dest_{ind}"] - r[f"origin_{ind}"]
            score += val * peso
            if delta > 0:
                motivi.append(f"{ind} ↑ (+{delta:.2f})")
            elif delta < 0:
                motivi.append(f"{ind} ↓ ({delta:.2f})")
            else:
                motivi.append(f"{ind} = ({val:.2f})")
        return pd.Series({"score": score, "motivi": ", ".join(motivi)})

    df_user[["score", "motivi"]] = df_user.apply(calcola_punteggio, axis=1)
    df_user["score_norm"] = MinMaxScaler().fit_transform(df_user[["score"]]) if df_user["score"].nunique() > 1 else 1.0

    if desidera:
        profilo = np.array([peso for peso in desidera.values()]).reshape(1, -1)
        matrice = df_user[[f"dest_{k}" for k in desidera.keys()]].values
        sim = cosine_similarity(profilo, matrice)[0]
    else:
        sim = np.zeros(len(df_user))

    df_user["similarity"] = sim
    df_user["score_finale"] = 0.5 * df_user["score_norm"] + 0.5 * df_user["similarity"]

    ranking = df_user.groupby("country_of_destination").agg({
        "score_finale": "mean",
        "motivi": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
    }).sort_values("score_finale", ascending=False).reset_index().head(top_n)
    return ranking

# === MOSTRA RISULTATI ===
if st.button("🔍 Scopri i paesi migliori"):
    risultato = consiglia_paesi(df, **user_input)
    st.subheader("📊 Paesi consigliati:")
    st.dataframe(risultato)

    st.subheader("📈 Punteggi Normalizzati")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(risultato["country_of_destination"], risultato["score_finale"], color="mediumseagreen")
    ax.invert_yaxis()
    ax.set_xlabel("Punteggio combinato")
    ax.set_title("Top Paesi Raccomandati")
    st.pyplot(fig)

# === CONFRONTO ===
st.subheader("📊 Confronta due Paesi")
col1, col2 = st.columns(2)
with col1:
    paese1 = st.selectbox("Paese 1", sorted(df["country_of_destination"].unique()))
with col2:
    paese2 = st.selectbox("Paese 2", sorted(df["country_of_destination"].unique()))

selected_inds = st.multiselect("Scegli indicatori da confrontare", indici, default=["Jobs", "Income"])
if st.button("📈 Confronta") and selected_inds:
    medie1 = df[df["country_of_destination"] == paese1][[f"dest_{i}" for i in selected_inds]].mean()
    medie2 = df[df["country_of_destination"] == paese2][[f"dest_{i}" for i in selected_inds]].mean()

    medie1.index = [i.replace("dest_", "") for i in medie1.index]
    medie2.index = [i.replace("dest_", "") for i in medie2.index]

    x = np.arange(len(selected_inds))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, medie1.values, width, label=paese1)
    ax.bar(x + width/2, medie2.values, width, label=paese2)
    ax.set_xticks(x)
    ax.set_xticklabels(selected_inds, rotation=45, ha='right')
    ax.set_title("Confronto tra Indicatori")
    ax.legend()
    st.pyplot(fig)

# === HASSE DIAGRAM ===
st.subheader("📌 Mostra Hasse")
try:
    import pygraphviz
    from networkx.drawing.nx_agraph import graphviz_layout
    
    var1 = st.selectbox("Variabile 1", valid_dest_cols, index=valid_dest_cols.index("dest_Jobs"))
    var2 = st.selectbox("Variabile 2", valid_dest_cols, index=valid_dest_cols.index("dest_Education"))
    var3 = st.selectbox("Variabile 3", valid_dest_cols, index=valid_dest_cols.index("dest_Safety"))
    color_metric = st.selectbox("Colora per", valid_dest_cols, index=valid_dest_cols.index("dest_Life satisfaction"))

    if st.button("📌 Mostra Hasse"):
        selected_vars = [var1, var2, var3]
        df_grouped = df.groupby("country_of_destination")[selected_vars + [color_metric]].mean()

        G = nx.DiGraph()
        countries = df_grouped.index.tolist()
        G.add_nodes_from(countries)

        for i in countries:
            for j in countries:
                if i == j: continue
                a = df_grouped.loc[i, selected_vars]
                b = df_grouped.loc[j, selected_vars]
                if dominates(a, b):
                    intermedi = [k for k in countries if dominates(df_grouped.loc[i, selected_vars], df_grouped.loc[k, selected_vars]) and dominates(df_grouped.loc[k, selected_vars], df_grouped.loc[j, selected_vars]) and k != i and k != j]
                    if not intermedi:
                        G.add_edge(i, j)

        pos = graphviz_layout(G, prog="dot")
        node_colors = df_grouped[color_metric].reindex(G.nodes()).values
        cmap = cm.get_cmap("Blues")
        norm = Normalize(vmin=min(node_colors), vmax=max(node_colors))

        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw(G, pos, node_color=node_colors, node_size=1000,
                cmap=cmap, edgecolors="black", with_labels=True,
                font_color="white", font_weight="bold", ax=ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label=color_metric)
        st.pyplot(fig)

except ImportError:
    st.error("Errore: pygraphviz non installato")

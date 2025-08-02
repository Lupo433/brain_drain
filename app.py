import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import Normalize
from matplotlib import cm
from networkx.drawing.nx_agraph import graphviz_layout

# === CONFIG ===
st.set_page_config(page_title="GoWhere - Brain Drain", layout="wide")
st.title("🌍 GoWhere - Brain Drain Analyzer")

# === LOAD DATA ===
df = pd.read_csv("dataset_final.csv")

# === INPUT UTENTE COMPLETO ===
st.sidebar.header("📥 Profilo Utente")
sex = st.sidebar.selectbox("Sesso", ["Male", "Female"])
origin_country = st.sidebar.selectbox("Paese di origine", sorted(df["country_of_birth"].unique()))

indici_disponibili = [
    "Education", "Jobs", "Income", "Safety", "Health", "Environment",
    "Civic engagement", "Accessiblity to services", "Housing",
    "Community", "Life satisfaction", "PR rating", "CL rating"
]

st.sidebar.subheader("❌ Cosa NON ti piace del tuo paese")
indici_da_migliorare = {}
for i in indici_disponibili:
    val = st.sidebar.slider(f"{i} ❌", 0.0, 10.0, 0.0, step=0.5)
    if val > 0:
        indici_da_migliorare[i] = val

st.sidebar.subheader("✅ Cosa DESIDERI nel nuovo paese")
indici_desiderati = {}
for i in indici_disponibili:
    val = st.sidebar.slider(f"{i} ✅", 0.0, 10.0, 0.0, step=0.5)
    if val > 0:
        indici_desiderati[i] = val

user_input = {
    "sex": sex,
    "origin_country": origin_country,
    "indici_da_migliorare": indici_da_migliorare,
    "indici_desiderati": indici_desiderati
}

# === FUNZIONE DI RACCOMANDAZIONE ===
def consiglia_paesi(df, user_input, top_n=5):
    orig = user_input["origin_country"]
    sex = user_input["sex"]
    migliora = user_input.get("indici_da_migliorare", {})
    desidera = user_input.get("indici_desiderati", {})

    df_user = df[(df["country_of_birth"] == orig) & (df["sex"] == sex)].copy()

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

    if df_user["score"].nunique() > 1:
        scaler = MinMaxScaler()
        df_user["score_norm"] = scaler.fit_transform(df_user[["score"]])
    else:
        df_user["score_norm"] = 1.0

    if desidera:
        profilo = np.array([peso for peso in desidera.values()]).reshape(1, -1)
        indici_sim = [f"dest_{k}" for k in desidera.keys()]
        matrice_dest = df_user[indici_sim].values
        similarità = cosine_similarity(profilo, matrice_dest)[0]
    else:
        similarità = np.zeros(len(df_user))

    df_user["similarity"] = similarità
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

# === OUTPUT ===
if st.button("🔍 Scopri i paesi migliori"):
    risultato = consiglia_paesi(df, user_input, top_n=5)
    st.subheader("🔝 Paesi consigliati")
    st.dataframe(risultato)

    fig, ax = plt.subplots()
    ax.barh(risultato["country_of_destination"], risultato["score_finale"], color="teal")
    ax.set_xlabel("Punteggio finale combinato")
    ax.set_title("Top Paesi Raccomandati")
    ax.invert_yaxis()
    st.pyplot(fig)

# === COMPARATORE PAESI ===
st.subheader("📊 Confronta due Paesi")
paesi = sorted(df["country_of_destination"].unique())
paese1 = st.selectbox("Paese 1", paesi)
paese2 = st.selectbox("Paese 2", paesi)
indici_sel = st.multiselect("Scegli indicatori da confrontare", indici_disponibili, default=["Jobs", "Income"])

if st.button("📈 Mostra Confronto") and indici_sel:
    medie1 = df[df["country_of_destination"] == paese1][[f"dest_{i}" for i in indici_sel]].mean()
    medie2 = df[df["country_of_destination"] == paese2][[f"dest_{i}" for i in indici_sel]].mean()

    x = range(len(indici_sel))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width/2 for i in x], medie1.values, width, label=paese1)
    ax.bar([i + width/2 for i in x], medie2.values, width, label=paese2)
    ax.set_xticks(x)
    ax.set_xticklabels(indici_sel, rotation=45, ha="right")
    ax.set_ylabel("Valore normalizzato")
    ax.set_title("Confronto Paesi")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    st.pyplot(fig)

# === HASSE DIAGRAM ===
st.subheader("📐 Relazioni tra Paesi: Hasse Diagram")
dest_columns = [col for col in df.columns if col.startswith("dest_")]
valid_dest_columns = [col for col in dest_columns if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().any()]

with st.expander("Visualizza Hasse Diagram"):
    var1 = st.selectbox("Variabile 1", valid_dest_columns, index=valid_dest_columns.index("dest_Jobs"))
    var2 = st.selectbox("Variabile 2", valid_dest_columns, index=valid_dest_columns.index("dest_Education"))
    var3 = st.selectbox("Variabile 3", valid_dest_columns, index=valid_dest_columns.index("dest_Safety"))
    color_metric = st.selectbox("Colora per", valid_dest_columns, index=valid_dest_columns.index("dest_Life satisfaction"))

    if st.button("📌 Mostra Hasse"):
        def dominates(a, b):
            return all(a >= b) and any(a > b)

        selected_vars = [var1, var2, var3]
        df_grouped = df.groupby("country_of_destination")[selected_vars + [color_metric]].mean()

        G = nx.DiGraph()
        countries = df_grouped.index.tolist()
        G.add_nodes_from(countries)

        for i in countries:
            for j in countries:
                if i == j:
                    continue
                a = df_grouped.loc[i, selected_vars]
                b = df_grouped.loc[j, selected_vars]
                if dominates(a, b):
                    intermedi = [k for k in countries if dominates(df_grouped.loc[i, selected_vars], df_grouped.loc[k, selected_vars])
                                 and dominates(df_grouped.loc[k, selected_vars], df_grouped.loc[j, selected_vars]) and k != i and k != j]
                    if not intermedi:
                        G.add_edge(i, j)

        try:
            pos = graphviz_layout(G, prog="dot")
        except:
            st.error("Errore: pygraphviz non installato")
            st.stop()

        color_vals = df_grouped[color_metric].to_dict()
        node_colors = [color_vals.get(n, 0.5) for n in G.nodes()]
        cmap = cm.get_cmap("Blues")
        norm = Normalize(vmin=min(node_colors), vmax=max(node_colors))
        node_sizes = [800 + 400 * G.out_degree(n) for n in G.nodes()]

        fig, ax = plt.subplots(figsize=(14, 10))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, cmap=cmap, ax=ax, edgecolors='black')

        for node in G.nodes():
            rgba = cmap(norm(color_vals.get(node, 0.5)))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            font_color = 'white' if luminance < 0.5 else 'black'
            nx.draw_networkx_labels(G, pos, labels={node: node}, font_color=font_color, font_size=9, ax=ax)

        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='gray', connectionstyle='arc3,rad=0.05')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label(color_metric)

        plt.title(f"Hasse Diagram – {var1}, {var2}, {var3}", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- DATASET ---
df = pd.read_csv("dataset_final.csv")

st.set_page_config(page_title="GoWhere - Brain Drain Analyzer", layout="centered")

# === HEADER ===
st.title("🌍 GoWhere - Brain Drain Analyzer")
st.markdown(
    """
    Questo strumento ti aiuta a scoprire i paesi migliori per le tue esigenze personali, confrontando indicatori
    chiave come lavoro, salute, ambiente e altro. Rispondi alle domande, scopri raccomandazioni personalizzate
    e confronta visivamente le opzioni disponibili.
    """
)

# === INDICI ===
indici_disponibili = [
    "Education", "Jobs", "Income", "Safety", "Health", "Environment",
    "Civic engagement", "Accessiblity to services", "Housing",
    "Community", "Life satisfaction", "PR rating", "CL rating"
]

istruzioni_indici = {
    "Education": "Qualità del sistema educativo.",
    "Jobs": "Opportunità di lavoro.",
    "Income": "Reddito medio.",
    "Safety": "Sicurezza personale.",
    "Health": "Sistema sanitario.",
    "Environment": "Qualità ambientale.",
    "Civic engagement": "Partecipazione civica.",
    "Accessiblity to services": "Accesso ai servizi.",
    "Housing": "Alloggi disponibili.",
    "Community": "Relazioni sociali.",
    "Life satisfaction": "Soddisfazione di vita.",
    "PR rating": "Libertà politica.",
    "CL rating": "Libertà civile."
}

# === PREFERENZE UTENTE ===
st.subheader("👤 Preferenze personali")
sex = st.selectbox("Sesso", ["Male", "Female"])
origin = st.selectbox("Paese di origine", sorted(df["country_of_birth"].unique()))

st.markdown("#### 1. Cosa NON ti piace del tuo paese?")
st.markdown("Seleziona gli aspetti del tuo paese attuale che vorresti migliorare trasferendoti. Più alto è il peso (0–10), più è importante per te allontanarti da quella caratteristica. Ad esempio: se senti che mancano opportunità di lavoro, seleziona "Jobs" con un peso alto.")
indici_da_migliorare = {}
for ind in indici_disponibili:
    col1, col2 = st.columns([2, 1])
    with col1:
        dislike = st.checkbox(f"{ind} ❌", key=f"dis_{ind}")
    with col2:
        if dislike:
            peso = st.slider(f"Peso {ind}", 0.0, 10.0, 5.0, 0.5, key=f"peso_dis_{ind}")
            indici_da_migliorare[ind] = peso

st.markdown("#### 2. Cosa DESIDERI trovare nel nuovo paese?")
st.markdown("Indica gli aspetti che cerchi attivamente nel nuovo paese, anche se non mancano necessariamente nel tuo. Seleziona ciò che ti attira, e assegna un peso più alto agli elementi più importanti per te. Ad esempio: vuoi un ambiente più sano o alta soddisfazione di vita? Seleziona "Environment" o "Life satisfaction". ")
indici_desiderati = {}
for ind in indici_disponibili:
    col1, col2 = st.columns([2, 1])
    with col1:
        like = st.checkbox(f"{ind} ✅", key=f"des_{ind}")
    with col2:
        if like:
            peso = st.slider(f"Peso {ind}", 0.0, 10.0, 5.0, 0.5, key=f"peso_des_{ind}")
            indici_desiderati[ind] = peso

# === RACCOMANDAZIONE ===
if st.button("🔍 Scopri i paesi migliori"):
    def consiglia_paesi(df, origin, sex, indici_da_migliorare, indici_desiderati, top_n=5):
        df_user = df[(df["country_of_birth"] == origin) & (df["sex"] == sex)].copy()

        def calcola_punteggio(r):
            score = 0
            motivi = []
            for ind, peso in indici_da_migliorare.items():
                delta = r[f"dest_{ind}"] - r[f"origin_{ind}"]
                contrib = delta * peso
                score += contrib
                motivi.append(f"{ind} {'↑' if delta>0 else '↓'} ({delta:.2f})")

            for ind, peso in indici_desiderati.items():
                val = r[f"dest_{ind}"]
                delta = val - r[f"origin_{ind}"]
                contrib = val * peso
                score += contrib
                motivi.append(f"{ind} {'↑' if delta>0 else '↓'} ({delta:.2f})")

            return pd.Series({"score": score, "motivi": ", ".join(motivi)})

        df_user[["score", "motivi"]] = df_user.apply(calcola_punteggio, axis=1)

        if df_user["score"].nunique() > 1:
            df_user["score_norm"] = MinMaxScaler().fit_transform(df_user[["score"]])
        else:
            df_user["score_norm"] = 1.0

        if indici_desiderati:
            profilo = np.array([peso for peso in indici_desiderati.values()]).reshape(1, -1)
            indici_sim = [f"dest_{k}" for k in indici_desiderati.keys()]
            matrice_dest = df_user[indici_sim].values
            sim = cosine_similarity(profilo, matrice_dest)[0]
        else:
            sim = np.zeros(len(df_user))

        df_user["similarity"] = sim
        df_user["score_finale"] = 0.5 * df_user["score_norm"] + 0.5 * df_user["similarity"]

        ranking = (
            df_user.groupby("country_of_destination")
            .agg({"score_finale": "mean", "motivi": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]})
            .sort_values("score_finale", ascending=False)
            .reset_index()
            .head(top_n)
        )
        return ranking

    risultato = consiglia_paesi(df, origin, sex, indici_da_migliorare, indici_desiderati)
    st.subheader("🔝 Paesi consigliati")
    st.dataframe(risultato)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(risultato["country_of_destination"], risultato["score_finale"], color="teal")
    ax.set_xlabel("Punteggio finale combinato")
    ax.set_title("Top Paesi Raccomandati")
    ax.invert_yaxis()
    st.pyplot(fig)

# === CONFRONTO DUE PAESI ===
st.subheader("📊 Confronta due Paesi")
paese1 = st.selectbox("Paese 1", sorted(df["country_of_destination"].unique()), key="p1")
paese2 = st.selectbox("Paese 2", sorted(df["country_of_destination"].unique()), key="p2")
indici_selezionati = st.multiselect("Scegli gli indici da confrontare", indici_disponibili, default=["Jobs", "Education"])

if indici_selezionati:
    medie1 = df[df["country_of_destination"] == paese1][[f"dest_{i}" for i in indici_selezionati]].mean()
    medie2 = df[df["country_of_destination"] == paese2][[f"dest_{i}" for i in indici_selezionati]].mean()
    x = range(len(indici_selezionati))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 0.2 for i in x], medie1.values, width=0.4, label=paese1)
    ax.bar([i + 0.2 for i in x], medie2.values, width=0.4, label=paese2)
    ax.set_xticks(x)
    ax.set_xticklabels(indici_selezionati, rotation=45, ha="right")
    ax.legend()
    st.pyplot(fig)

# === HASSE DIAGRAM ===
st.subheader("📈 Visualizza relazioni tra Paesi")
with st.expander("Mostra diagramma Hasse personalizzato"):
    dest_cols = [col for col in df.columns if col.startswith("dest_")]
    var1 = st.selectbox("Variabile 1", dest_cols)
    var2 = st.selectbox("Variabile 2", dest_cols)
    var3 = st.selectbox("Variabile 3", dest_cols)
    color_metric = st.selectbox("Colora per", dest_cols)

    def dominates(a, b):
        return all(a >= b) and any(a > b)

    if st.button("📌 Mostra Hasse"):
        st.info("Costruzione grafo...")
        selected = [var1, var2, var3]
        df_grouped = df.groupby("country_of_destination")[selected + [color_metric]].mean()
        G = nx.DiGraph()
        countries = df_grouped.index.tolist()
        for i in countries:
            for j in countries:
                if i == j: continue
                a, b = df_grouped.loc[i, selected], df_grouped.loc[j, selected]
                if dominates(a, b):
                    intermedi = [k for k in countries if dominates(df_grouped.loc[i, selected], df_grouped.loc[k, selected]) and dominates(df_grouped.loc[k, selected], df_grouped.loc[j, selected]) and k != i and k != j]
                    if not intermedi:
                        G.add_edge(i, j)

        pos = nx.spring_layout(G, seed=42)
        color_vals = df_grouped[color_metric].to_dict()
        node_colors = [color_vals.get(n, 0.5) for n in G.nodes()]
        cmap = plt.cm.Blues
        norm = plt.Normalize(min(node_colors), max(node_colors))
        node_sizes = [800 + 400 * G.out_degree(n) for n in G.nodes()]

        fig, ax = plt.subplots(figsize=(14, 10))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, cmap=cmap, ax=ax, edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', arrowsize=20, edge_color='gray')

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
        cbar.set_label(color_metric)

        plt.title("Hasse Diagram", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        st.pyplot(fig)

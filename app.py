import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
import matplotlib.cm as cm

# --- FUNZIONE ---
def consiglia_paesi(df, user_input, top_n=5):
    orig = user_input["origin_country"]
    sex = user_input["sex"]
    migliora = user_input["indici_da_migliorare"]
    desidera = user_input["indici_desiderati"]

    df_user = df[(df["country_of_birth"] == orig) & (df["sex"] == sex)].copy()

    def calcola_punteggio(r):
        score = 0
        motivi = []
        for ind, peso in migliora.items():
            delta = r[f"dest_{ind}"] - r[f"origin_{ind}"]
            contrib = max(delta, 0) * peso
            score += contrib
            if delta > 0:
                motivi.append(f"{ind} ↑ (+{delta:.2f})")
        for ind, peso in desidera.items():
            val = r[f"dest_{ind}"]
            contrib = val * peso
            score += contrib
            motivi.append(f"{ind} = {val:.2f}")
        return pd.Series({"score": score, "motivi": ", ".join(motivi)})

    df_user[["score", "motivi"]] = df_user.apply(calcola_punteggio, axis=1)
    min_score = df_user["score"].min()
    max_score = df_user["score"].max()
    df_user["score_norm"] = (df_user["score"] - min_score) / (max_score - min_score + 1e-9)

    ranking = (
        df_user.groupby("country_of_destination")
        .agg({
            "score_norm": "mean",
            "motivi": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
        })
        .sort_values("score_norm", ascending=False)
        .reset_index()
        .head(top_n)
    )
    return ranking

# Funzione Hasse Diagram
def build_hasse(df, selected_vars, color_metric):
    df_grouped = df.groupby("country_of_destination")[selected_vars + [color_metric]].mean()
    G = nx.DiGraph()
    countries = df_grouped.index.tolist()
    G.add_nodes_from(countries)

    def dominates(a, b):
        return all(a >= b) and any(a > b)

    for i in countries:
        for j in countries:
            if i == j:
                continue
            a = df_grouped.loc[i, [var1, var2, var3]]
            b = df_grouped.loc[j, [var1, var2, var3]]
            if dominates(a, b):
                intermedi = [k for k in countries if dominates(df_grouped.loc[i, [var1, var2, var3]], df_grouped.loc[k, [var1, var2, var3]])
                             and dominates(df_grouped.loc[k, [var1, var2, var3]], df_grouped.loc[j, [var1, var2, var3]]) and k != i and k != j]
                if not intermedi:
                    G.add_edge(i, j)

    try:
        pos = nx.nx_pydot.graphviz_layout(G, prog="dot")
    except:
        pos = nx.spring_layout(G)

    fig, ax = plt.subplots(figsize=(14, 10))
    node_colors = [df_grouped.loc[n, color_metric] for n in G.nodes()]
    node_sizes = [800 + 400 * G.out_degree(n) for n in G.nodes()]
    norm = Normalize(vmin=min(node_colors), vmax=max(node_colors))
    cmap = cm.get_cmap('Blues')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=cmap, node_size=node_sizes, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowsize=25, edge_color="gray")
    for node in G.nodes():
        color = cmap(norm(df_grouped.loc[node, color_metric]))
        luminance = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        font_color = 'white' if luminance < 0.5 else 'black'
        nx.draw_networkx_labels(G, pos, labels={node: node}, font_color=font_color, font_size=8, ax=ax)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label(color_metric)

    ax.set_title(f"Hasse Diagram – {var1}, {var2}, {var3}", fontsize=14)
    ax.axis('off')
    st.pyplot(fig)

# --- INTERFACCIA APP ---
st.set_page_config(page_title="GoWhere", layout="wide")
st.title("🌍 GoWhere - Trova il tuo paese ideale")
st.markdown("Rispondi a poche domande e scopri in quali paesi potresti vivere meglio!")

# Input
sex = st.selectbox("Qual è il tuo sesso?", ["Male", "Female"])
origin_country = st.text_input("Inserisci il tuo paese di origine (es. ITA):", "ITA")

st.subheader("Cosa vuoi migliorare?")
income = st.slider("Reddito", 0, 5, 3)
jobs = st.slider("Opportunità di lavoro", 0, 5, 3)
safety = st.slider("Sicurezza", 0, 5, 3)

st.subheader("Cosa ti interessa di più?")
life = st.slider("Soddisfazione di vita", 0, 5, 3)
env = st.slider("Ambiente", 0, 5, 2)

# Bottone
if st.button("🔍 Scopri i paesi migliori"):
    try:
        df = pd.read_csv("dataset_final.csv")
        user_input = {
            "sex": sex,
            "origin_country": origin_country,
            "indici_da_migliorare": {
                "Income": income,
                "Safety": safety,
                "Jobs": jobs
            },
            "indici_desiderati": {
                "Life satisfaction": life,
                "Environment": env
            }
        }
        risultato = consiglia_paesi(df, user_input)
        st.subheader("🔝 Paesi consigliati:")
        st.dataframe(risultato)

        st.subheader("📊 Punteggi Normalizzati")
        fig, ax = plt.subplots()
        ax.barh(risultato["country_of_destination"], risultato["score_norm"], color="mediumseagreen")
        ax.set_xlabel("Punteggio normalizzato (0–1)")
        ax.set_title("Top Paesi Consigliati")
        ax.invert_yaxis()
        st.pyplot(fig)

        # --- OPZIONE: Visualizza Hasse Diagram personalizzato ---
        st.subheader("📈 Relazioni tra Paesi (Diagramma Hasse)")
        st.markdown("Puoi confrontare i paesi su 2-4 indicatori a tua scelta. I nodi sono i paesi, e le frecce mostrano dominanze multi-indicatore.")

        if st.checkbox("✅ Visualizza Hasse Diagram personalizzato"):
            dest_columns = [col for col in df.columns if col.startswith("dest_")]

            hasse_vars = st.multiselect(
                "Seleziona da 2 a 4 indicatori per confrontare i Paesi",
                options=dest_columns,
                default=["dest_Jobs", "dest_Education", "dest_Safety"]
            )

            color_metric = st.selectbox(
                "Colore dei nodi in base a:",
                options=dest_columns,
                index=dest_columns.index("dest_Life satisfaction") if "dest_Life satisfaction" in dest_columns else 0
            )

            if len(hasse_vars) >= 2:
                st.markdown("✅ Generazione del grafo in corso...")
                build_hasse(df, hasse_vars, color_metric)
            else:
                st.warning("Seleziona almeno 2 variabili per creare il grafo.")



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib import cm

# === FUNZIONE DI CONSIGLIO PAESI ===
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

# === FUNZIONE DIAGRAMMA HASSE ===
def dominates(a, b):
    return all(a >= b) and any(a > b)

def hierarchy_pos_multiple_roots(G):
    def hierarchy_pos(G, root, width=1., vert_gap=0.5, vert_loc=0, xcenter=0.5, pos=None):
        if pos is None:
            pos = {}
        pos[root] = (xcenter, vert_loc)
        children = list(G.successors(root))
        if children:
            dx = width / len(children)
            nextx = xcenter - width / 2 + dx / 2
            for child in children:
                pos = hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                    vert_loc=vert_loc - vert_gap, xcenter=nextx, pos=pos)
                nextx += dx
        return pos

    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    full_pos = {}
    spacing = 2.5 / max(len(roots), 1)
    for i, root in enumerate(roots):
        sub_pos = hierarchy_pos(G, root, width=2.0, xcenter=i * spacing + spacing / 2)
        full_pos.update(sub_pos)
    return full_pos

def build_hasse(df, selected_vars, color_metric):
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

    pos = hierarchy_pos_multiple_roots(G)
    color_vals = df_grouped[color_metric].to_dict()
    node_colors = [color_vals.get(n, 0.5) for n in G.nodes()]
    cmap = cm.get_cmap('Blues')
    norm = Normalize(vmin=min(node_colors), vmax=max(node_colors))
    node_sizes = [800 + 400 * G.out_degree(n) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                           cmap=cmap, ax=ax, edgecolors="black")

    for node in G.nodes():
        rgba = cmap(norm(color_vals.get(node, 0.5)))
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        font_color = 'white' if luminance < 0.5 else 'black'
        nx.draw_networkx_labels(G, pos, labels={node: node}, font_color=font_color, font_size=9, ax=ax)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="gray", arrows=True, arrowsize=20, width=1.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label(color_metric)
    plt.title(f"Hasse Diagram – {', '.join(selected_vars)}", fontsize=14)
    plt.axis("off")
    st.pyplot(fig)

# === INTERFACCIA ===
st.set_page_config(page_title="GoWhere", layout="wide")
st.title("🌍 GoWhere - Trova il tuo paese ideale")
st.markdown("Rispondi a poche domande e scopri in quali paesi potresti vivere meglio!")

# Input utente
sex = st.selectbox("Qual è il tuo sesso?", ["Male", "Female"])
origin_country = st.text_input("Inserisci il tuo paese di origine (es. ITA):", "ITA")

df = pd.read_csv("dataset_final.csv")
indici = [col.replace("dest_", "") for col in df.columns if col.startswith("dest_")]

st.subheader("Cosa vuoi migliorare?")
indici_da_migliorare = {}
for ind in indici:
    indici_da_migliorare[ind] = st.slider(ind, 0, 5, 3, key="migliora_" + ind)

st.subheader("Cosa ti interessa di più?")
indici_desiderati = {}
for ind in indici:
    indici_desiderati[ind] = st.slider(ind, 0, 5, 2, key="desidera_" + ind)

if st.button("🔍 Scopri i paesi migliori"):
    user_input = {
        "sex": sex,
        "origin_country": origin_country,
        "indici_da_migliorare": indici_da_migliorare,
        "indici_desiderati": indici_desiderati
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

# Hasse diagram
st.subheader("📈 Visualizza relazioni tra Paesi (opzionale)")
with st.expander("Mostra diagramma Hasse personalizzato", expanded=True):
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
        build_hasse(df, hasse_vars, color_metric)
    else:
        st.info("Seleziona almeno 2 variabili per creare il grafo.")

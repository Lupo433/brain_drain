import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# === LOAD DATA ===
df = pd.read_csv("dataset_final.csv")

st.set_page_config(page_title="GoWhere - Brain Drain Analyzer", layout="centered")

# === TITLE ===
st.title("🌍 2222 GoWhere - Brain Drain Analyzer")
st.markdown("""
This tool helps you find the best countries based on your personal preferences,
comparing key factors like jobs, safety, health, and more. Answer a few questions
to get personalized recommendations and visualize the best destinations.
""")

# === INDICATORS ===
indicators = [
    "Education", "Jobs", "Income", "Safety", "Health", "Environment",
    "Civic engagement", "Accessiblity to services", "Housing",
    "Community", "Life satisfaction", "PR rating", "CL rating"
]

indicator_help = {
    "Education": "Quality of education system",
    "Jobs": "Employment opportunities",
    "Income": "Average income level",
    "Safety": "Personal safety and crime rates",
    "Health": "Healthcare system and services",
    "Environment": "Air quality, pollution, green areas",
    "Civic engagement": "Citizen participation and democracy",
    "Accessiblity to services": "Access to basic services like transport and internet",
    "Housing": "Availability and affordability of housing",
    "Community": "Social connections and trust",
    "Life satisfaction": "General well-being and happiness",
    "PR rating": "Political rights and freedoms",
    "CL rating": "Civil liberties and protections"
}

# === USER PREFERENCES ===
st.subheader("🧭 Customize your preferences")

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Select your gender", ["Male", "Female"], help="Your gender may influence preferences and migration motivations.")
with col2:
    origin = st.selectbox("Select your country of origin", sorted(df["country_of_birth"].unique()), help="The country you currently live in.")

st.markdown("### ❌ What do you want to improve in your current country?")
st.caption("Select things you’d like to escape or improve and assign importance (0–10).")
indices_to_improve = {}
cols = st.columns(2)
for i, ind in enumerate(indicators):
    with cols[i % 2]:
        if st.checkbox(f"{ind}", key=f"imp_{ind}", help=indicator_help[ind]):
            weight = st.slider(f"Weight for {ind}", 0.0, 10.0, 5.0, 0.5, key=f"w_imp_{ind}")
            indices_to_improve[ind] = weight

st.markdown("### ✅ What do you desire in a new country?")
st.caption("Select aspects that matter to you even if they're already good at home.")
indices_desired = {}
cols2 = st.columns(2)
for i, ind in enumerate(indicators):
    with cols2[i % 2]:
        if st.checkbox(f"{ind}", key=f"des_{ind}", help=indicator_help[ind]):
            weight = st.slider(f"Weight for {ind}", 0.0, 10.0, 5.0, 0.5, key=f"w_des_{ind}")
            indices_desired[ind] = weight

# === RECOMMENDATION ENGINE ===
if st.button("🔍 Discover best countries"):
    def recommend_countries(df, origin, sex, to_improve, desired, top_n=5):
        df_user = df[(df["country_of_birth"] == origin) & (df["sex"] == sex)].copy()

        def compute_score(r):
            score = 0
            reasons = []
            for ind, weight in to_improve.items():
                delta = r[f"dest_{ind}"] - r[f"origin_{ind}"]
                score += delta * weight
                reasons.append(f"{ind} {'↑' if delta>0 else '↓'} ({delta:.2f})")
            for ind, weight in desired.items():
                val = r[f"dest_{ind}"]
                delta = val - r[f"origin_{ind}"]
                score += val * weight
                reasons.append(f"{ind} {'↑' if delta>0 else '↓'} ({delta:.2f})")
            return pd.Series({"score": score, "reasons": ", ".join(reasons)})

        df_user[["score", "reasons"]] = df_user.apply(compute_score, axis=1)
        df_user["score_norm"] = (
            MinMaxScaler().fit_transform(df_user[["score"]])
            if df_user["score"].nunique() > 1 else 1.0
        )

        if desired:
            profile = np.array(list(desired.values())).reshape(1, -1)
            sim_indices = [f"dest_{k}" for k in desired.keys()]
            sim_matrix = df_user[sim_indices].values
            sim = cosine_similarity(profile, sim_matrix)[0]
        else:
            sim = np.zeros(len(df_user))

        df_user["similarity"] = sim
        df_user["final_score"] = 0.5 * df_user["score_norm"] + 0.5 * df_user["similarity"]

        top_countries = (
            df_user.groupby("country_of_destination")
            .agg({
                "final_score": "mean",
                "reasons": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
            })
            .sort_values("final_score", ascending=False)
            .reset_index()
            .head(top_n)
        )
        return top_countries

    result = recommend_countries(df, origin, sex, indices_to_improve, indices_desired)

    # === DISPLAY TABULAR RESULTS (with word wrapping) ===
    if not result.empty:
        st.markdown("### 🏆 Recommended Countries")
        st.markdown("""
        Here are the countries that best match your preferences. You can review the reasoning behind the score for each destination.
        """)
    
        for idx, row in result.iterrows():
            country = row['country_of_destination']
            score = row['final_score']
            reasons = row["reasons"].split(", ")
        
            # HTML Card
            card_html = f"""
            <div style="background-color:#f9f9f9; padding:20px; border-radius:10px; border:1px solid #ddd; margin-bottom:20px;">
                <h3 style="margin-bottom:10px;">{idx+1}. <b>{country}</b> — <span style='background-color:#e8f5e9; color:#2e7d32; padding:4px 10px; border-radius:5px; font-family:monospace;'>{score:.4f}</span></h3>
                <ul style="padding-left:20px; line-height:1.6;">
            """
        
            for reason in reasons:
                card_html += f"<li>{reason}</li>"
        
            card_html += "</ul></div>"
        
            st.markdown(card_html, unsafe_allow_html=True)

    
        st.markdown("### 📊 Visualization of top scores")
        st.markdown("This chart shows how strongly each recommended country matches your personal preferences.")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.barh(result["country_of_destination"], result["final_score"], color="teal", height=0.4)
        ax.set_xlabel("Final combined score")
        ax.set_title("Top Recommended Countries")
        ax.invert_yaxis()
        st.pyplot(fig)

# === COMPARE COUNTRIES ===
st.subheader("📊 Compare two Countries")

st.markdown(
    "Use the tool below to visually compare how two countries perform on selected key indicators. "
    "This can help you understand the relative strengths and weaknesses of each destination based on your priorities."
)

p1 = st.selectbox("Country 1", sorted(df["country_of_destination"].unique()), key="p1")
p2 = st.selectbox("Country 2", sorted(df["country_of_destination"].unique()), key="p2")
selected_ind = st.multiselect("Select indicators to compare", indicators, default=["Jobs", "Education"])

if selected_ind:
    avg1 = df[df["country_of_destination"] == p1][[f"dest_{i}" for i in selected_ind]].mean()
    avg2 = df[df["country_of_destination"] == p2][[f"dest_{i}" for i in selected_ind]].mean()
    x = range(len(selected_ind))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 0.2 for i in x], avg1.values, width=0.4, label=p1)
    ax.bar([i + 0.2 for i in x], avg2.values, width=0.4, label=p2)
    ax.set_xticks(x)
    ax.set_xticklabels(selected_ind, rotation=45, ha="right")
    ax.legend()
    st.pyplot(fig)


import plotly.express as px

# === GEOGRAPHIC HEATMAP (OECD Only) ===
st.subheader("🗺️ Geographic Heatmap of OECD Countries")

# Lista ISO-3 dei paesi membri OCSE
oecd_countries = [
    "AUS", "AUT", "BEL", "CAN", "CHE", "CHL", "COL", "CZE", "DNK", "EST",
    "FIN", "FRA", "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "JPN",
    "KOR", "LVA", "LTU", "LUX", "MEX", "NLD", "NZL", "NOR", "POL", "PRT",
    "SVK", "SVN", "ESP", "SWE", "TUR", "GBR", "USA"
]

# Seleziona la variabile per la colorazione e altre per il tooltip
st.markdown("Seleziona la variabile principale per colorare la mappa e altre da mostrare nel tooltip.")
all_vars = [f"dest_{ind}" for ind in indicators]
color_var = st.selectbox("🌡️ Variabile da colorare", all_vars, index=all_vars.index("dest_Safety"))
tooltip_vars = st.multiselect("🧾 Variabili nel tooltip", all_vars, default=["dest_Jobs", "dest_Income", "dest_Health"])

# Crea il DataFrame medio per paese e filtra i paesi OECD
df_grouped = df.groupby("country_of_destination").mean(numeric_only=True).reset_index()
df_grouped = df_grouped[df_grouped["country_of_destination"].isin(oecd_countries)]
df_grouped["iso_alpha"] = df_grouped["country_of_destination"]

# Costruzione del tooltip
df_grouped["tooltip"] = df_grouped.apply(
    lambda row: "<br>".join([f"{v.replace('dest_', '')}: {row[v]:.2f}" for v in tooltip_vars]),
    axis=1
)

# Creazione mappa Plotly
fig = px.choropleth(
    df_grouped,
    locations="iso_alpha",
    locationmode="ISO-3",
    color=color_var,
    hover_name="country_of_destination",
    hover_data={"tooltip": True, "iso_alpha": False, color_var: True},
    color_continuous_scale="Blues",
    title=f"{color_var.replace('dest_', '')} across OECD Countries"
)

fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>%{customdata[0]}")
fig.update_layout(
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
)

st.plotly_chart(fig, use_container_width=True)


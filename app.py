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

st.markdown("""
<style>
    /* === BASE DARK THEME === */
    body, .stApp {
        background-color: #121212;
        color: white;
    }

    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stText, .stSubheader, .stCaption,
    .stCheckbox > label > div, .stSlider label, .stRadio label,
    label[data-testid="stMarkdownContainer"],
    div[data-testid="stMarkdownContainer"] p {
        color: white !important;
    }

    /* Dropdowns e slider */
    .css-1v0mbdj, .css-1cpxqw2 {
        background-color: #2e2e2e !important;
        color: white !important;
    }
    .css-1cpxqw2:hover {
        border-color: #ffffff !important;
    }

    /* Nasconde completamente le icone di ancoraggio accanto ai titoli */
    a[href^="#"] {
        display: none !important;
    }

    /* Pulsanti */
    .stButton>button {
        color: white;
        background-color: #333333;
    }
    .stButton>button:hover {
        background-color: #444444;
    }

    /* === TOOLTIP === */

    /* Icona punto interrogativo */
    div[data-testid="stTooltipIcon"] svg,
    svg[data-testid="icon-help"] {
        stroke: #cccccc !important;
        fill: #000000 !important;
    }

    /* Box tooltip */
    div[role="tooltip"] {
        background-color: #ffffff !important;
        color: black !important;
        border-radius: 6px !important;
        padding: 8px 10px !important;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.2);
        z-index: 9999 !important;
        opacity: 1 !important;
    }

    /* TESTO tooltip - include qualsiasi cosa dentro */
    div[role="tooltip"] *,
    div[role="tooltip"] label,
    div[role="tooltip"] span,
    div[role="tooltip"] p {
        color: black !important;
        font-weight: 500 !important;
    }

    /* === ICONE ANCORAGGIO === */
    a[href^="#"] svg {
        stroke: #b3b3b3 !important;
        background-color: #000 !important;
        border-radius: 5px;
        padding: 3px;
        opacity: 1 !important;
    }
    a[href^="#"] {
        color: inherit !important;
    }

    /* === RIPRISTINO 3 PUNTINI (menu header Streamlit) === */
    button[data-testid="stBaseButton-headerNoPadding"] {
        all: unset !important;  /* Rimuove padding, bg, border, ecc */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
    
    /* SVG interno (tre puntini) */
    button[data-testid="stBaseButton-headerNoPadding"] svg {
        width: 20px !important;
        height: 20px !important;
        stroke: #000 !important;
        fill: #000 !important;
        opacity: 1 !important;
        background: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }


    /* === RISULTATI (cards) === */
    div[data-testid="stMarkdownContainer"] h3 {
        color: white !important;
    }
    div[data-testid="stMarkdownContainer"] ul {
        color: white !important;
    }
    div[data-testid="stMarkdownContainer"] h3 span {
        background-color: #e8f5e9 !important;
        color: #2e7d32 !important;
        font-weight: bold;
    }
    div[data-testid="stMarkdownContainer"] > div {
        background-color: #1e1e1e !important;
    }

</style>
""", unsafe_allow_html=True)





# === TITLE ===
st.title("🌍 GoWhere - Brain Drain Analyzer")
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
            reasons = list(dict.fromkeys(row["reasons"].split(", ")))
        
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


# === CLUSTERING ===
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.subheader("🔍 Country clusters by indicators")

st.markdown("""
This tool allows you to explore how countries group together based on selected development indicators.  
By choosing at least two indicators (e.g., Education, Health, Income), the app uses **clustering and PCA (Principal Component Analysis)** to visually position similar countries together in a 2D space.

Each cluster groups countries with comparable performance on the selected metrics, and the legend provides a summary of each cluster’s strengths and weaknesses.  
Use this visualization to identify patterns, outliers, or similarities across countries in terms of well-being, opportunity, and quality of life.
""")

indices = indicators

selected = st.multiselect(
    "Select at least two indices to group countries",
    options=indices,
    default=["Education", "Income"]
)

if len(selected) < 2:
    st.warning("⚠️ Select at least two indexes to continue.")
else:
    try:
        cols = [f"dest_{i}" for i in selected]
        df_media = df.groupby("country_of_destination")[cols].mean().reset_index()
        df_media.dropna(inplace=True)

        if df_media.empty:
            st.error("⚠️ No valid data available.")
        else:
            X = StandardScaler().fit_transform(df_media[cols])

            # === Coordinate PCA oppure originali ===
            if X.shape[1] > 2:
                X_pca = PCA(n_components=2).fit_transform(X)
                df_media["X"] = X_pca[:, 0]
                df_media["Y"] = X_pca[:, 1]
                x_label, y_label = "Principal Component 1", "Principal Component 2"
            else:
                var1, var2 = selected[0], selected[1]
                df_media["X"] = X[:, 0]
                df_media["Y"] = X[:, 1]
                x_label, y_label = var1, var2

            # === Silhouette score for optimal cluster count ===
            silhouette_scores = []
            cluster_range = range(2, min(10, len(df_media)))
            for k in cluster_range:
                km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                labels = km.fit_predict(X)
                score = silhouette_score(X, labels)
                silhouette_scores.append((k, score))

            best_k = max(silhouette_scores, key=lambda x: x[1])[0]
            kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
            df_media["Cluster"] = kmeans.fit_predict(X)

            # === Cluster label description ===
            cluster_summary = df_media.groupby("Cluster")[cols].mean()
            cluster_labels = {}
            for cluster_id, row in cluster_summary.iterrows():
                high = [col.replace("dest_", "") for col, val in row.items() if val >= 0.7]
                medium = [col.replace("dest_", "") for col, val in row.items() if 0.4 <= val < 0.7]
                low = [col.replace("dest_", "") for col, val in row.items() if val < 0.4]
                avg = row.mean()
                label = f"Cluster {cluster_id}"
                if high:
                    label += f"  | High: {', '.join(high)}"
                if medium:
                    label += f"  | Medium: {', '.join(medium)}"
                if low:
                    label += f"  | Low: {', '.join(low)}"
                label += f"  | Avg: {avg:.2f}"
                cluster_labels[cluster_id] = label

            df_media["Cluster_label"] = df_media["Cluster"].map(cluster_labels)
            df_media["text"] = df_media["country_of_destination"]

            hover_data = {
                "country_of_destination": True,
                **{col: True for col in cols}
            }

            # === Adjust size dynamically like Colab ===
            n_clusters = df_media["Cluster_label"].nunique()
            fig_height = 600 + (n_clusters * 25)
            bottom_margin = 100 if n_clusters <= 5 else 80 + n_clusters * 10

            fig = px.scatter(
                df_media, x="X", y="Y",
                color="Cluster_label",
                text="text",
                title="🌍 Country Cluster – Based on selected indicators",
                labels={"X": x_label, "Y": y_label},
                hover_data=hover_data,
                width=1000, height=fig_height
            )

            fig.update_traces(textposition="top center", marker=dict(size=9))

            x_min = df_media["X"].min() - 1
            x_max = df_media["X"].max() + 1
            fig.update_xaxes(tick0=round(x_min), dtick=0.5, range=[x_min, x_max])

            y_min = df_media["Y"].min() - 1
            y_max = df_media["Y"].max() + 1
            fig.update_yaxes(tick0=round(y_min), dtick=0.5, range=[y_min, y_max])

            fig.update_layout(
                legend_title_text="Cluster",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                ),
                margin=dict(l=50, r=50, t=60, b=bottom_margin)
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error while generating cluster: {str(e)}")

# === CONCLUSIONE FINALE ===
st.markdown("## 🧭 Final Thoughts – Beyond the Data")

st.markdown("""
You've taken a deep dive into the factors that matter—like **Education**, **Health**, **Income**, **Safety**, and more—and identified top destination countries based on your personal profile.  
With visual clustering and score breakdowns, this app helps illuminate not just **where** but **why** some countries match your priorities better than others.
""")

st.markdown("""
#### But let's zoom out and reflect on the bigger picture: *brain drain vs. brain gain*

Research increasingly shows that while skilled migration may seem like a loss for origin countries, it often leads to **net gains** when the right systems are in place:
- 🧠 Countries with **flexible education & training systems** can adapt to talent outflows by upskilling more citizens—triggering a broader **brain gain** effect.
- 🌍 **Diaspora networks**, return migration, and remittances fuel entrepreneurship, knowledge exchange, and trade, boosting development at home.

> Source: [Yale EGC – Brain Drain or Brain Gain?](https://egc.yale.edu/research/brain-drain-or-brain-gain-new-research-identifies-more-nuanced-story-about-skilled-migration?)
""")

st.markdown("""
---

### ✔️ In Short:
- Your insights help **maximize the benefits** of skilled migration by focusing on opportunities—not just destinations.
- Whether you're considering **temporary relocation**, **return plans**, or **career exploration**, this tool empowers informed decision-making.
- For policymakers, it highlights the importance of investing in education, mobility, and digital infrastructure to **turn migration into a win‑win strategy**.

---

### Thank You for Exploring with GoWhere 🌍

We hope this app has given you confidence and clarity. Let the data guide your journey—and may your next destination be one that aligns with your goals, values, and potential.

If you'd like to explore more variables or update your preferences, just scroll up and try again.
""")


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
    "Civic engagement", "Accessibility to services", "Housing",
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
    "Accessibility to services": "Access to basic services like transport and internet",
    "Housing": "Availability and affordability of housing",
    "Community": "Social connections and trust",
    "Life satisfaction": "General well-being and happiness",
    "PR rating": "Political rights and freedoms",
    "CL rating": "Civil liberties and protections"
}

# === USER PREFERENCES ===
st.subheader("🧭 Customize your preferences")

# Mapping for all countries (origin + destination)
country_map_df = pd.concat([
    df[["country_of_birth", "Country_origin"]].rename(columns={"country_of_birth": "code", "Country_origin": "name"}),
    df[["country_of_destination", "Country_dest"]].rename(columns={"country_of_destination": "code", "Country_dest": "name"})
]).drop_duplicates().sort_values("name")

country_display_map = {f"{row['code']} - {row['name']}": row["code"] for _, row in country_map_df.iterrows()}
code_to_name = {row["code"]: row["name"] for _, row in country_map_df.iterrows()}

col1, col2 = st.columns(2)
with col1:
    sex = st.selectbox("Select your gender", ["Male", "Female"], help="Your gender may influence preferences and migration motivations.")
with col2:
    origin_label = st.selectbox("Select your country of origin", list(country_display_map.keys()), help="This is the country you currently live in or want to compare against.")
    origin = country_display_map[origin_label]

st.markdown("### ❌ What aspects of your current country do you dislike?")
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
        origin_values = (
            df[(df["country_of_birth"] == origin) & (df["sex"] == sex)]
            [[f"origin_{ind}" for ind in indicators]]
            .mean()
        )

        dest_values = (
            df[(df["sex"] == sex)]
            .groupby("country_of_destination")[[f"dest_{ind}" for ind in indicators]]
            .mean()
            .copy()
        )


        flows = (
            df[(df["sex"] == sex) & (df["country_of_birth"] == origin)]
            .groupby("country_of_destination")["number"]
            .sum()
        )
        flows = flows.reindex(dest_values.index).fillna(0.0)
        flows = np.log1p(flows)  # log per stabilizzare

        
        rows = []
        for dest_code in dest_values.index:
            row = {"country_of_destination": dest_code}
            for ind in indicators:
                row[f"origin_{ind}"] = origin_values.get(f"origin_{ind}", np.nan)
                row[f"dest_{ind}"] = dest_values.loc[dest_code, f"dest_{ind}"]
                row["number"] = flows.get(dest_code, 0.0)
            rows.append(row)

        df_user = pd.DataFrame(rows).dropna()

        def compute_score(r):
            score = 0
            reasons = []
            for ind, weight in to_improve.items():
                delta = r[f"dest_{ind}"] - r[f"origin_{ind}"]
                score += delta * weight
                if delta >= 0.01:
                    reasons.append(f"{ind} ↑ (+{delta:.2f})")
            for ind, weight in desired.items():
                val = r[f"dest_{ind}"]
                delta = val - r[f"origin_{ind}"]
                score += val * weight
                if delta >= 0.01:
                    reasons.append(f"{ind} ↑ (+{delta:.2f})")
            return pd.Series({"score": score, "reasons": reasons})

        df_user[["score", "reasons"]] = df_user.apply(compute_score, axis=1)

        df_user["score_norm"] = (
            MinMaxScaler().fit_transform(df_user[["score"]])
            if df_user["score"].nunique() > 1 else 1.0
        )

        if desired:
            profile = np.array(list(desired.values())).reshape(1, -1)
            sim_cols = [f"dest_{k}" for k in desired]
            sim_matrix = df_user[sim_cols].values
            sim = cosine_similarity(profile, sim_matrix)[0]
        else:
            sim = np.zeros(len(df_user))

        df_user["number_norm"] = (
            MinMaxScaler().fit_transform(df_user[["number"]])
            if df_user["number"].nunique() > 1 else 1.0
        )


        df_user["similarity"] = sim
        df_user["final_score"] = (
            0.4 * df_user["score_norm"] +
            0.4 * df_user["similarity"] +
            0.2 * df_user["number_norm"]
        )


        def merge_reasons(series):
            flat = [item for sublist in series for item in sublist]
            return sorted(set(flat))

        result = (
            df_user.groupby("country_of_destination")
            .agg({
                "final_score": "mean",
                "reasons": merge_reasons
            })
            .sort_values("final_score", ascending=False)
            .reset_index()
            .head(top_n)
        )

        return result

    result = recommend_countries(df, origin, sex, indices_to_improve, indices_desired)

    if not result.empty:
        st.markdown("### 🏆 Recommended Countries")
        st.markdown("Here are the countries that best match your preferences:")

        for idx, row in result.iterrows():
            country_acronym = row['country_of_destination']
            country = code_to_name.get(country_acronym, country_acronym)
            score = row['final_score']
            reasons = row["reasons"]
        
            card_html = f"""
            <div style="background-color:#f9f9f9; padding:20px; border-radius:10px; border:1px solid #ddd; margin-bottom:20px;">
                <h3 style="margin-bottom:10px;">{idx+1}. <b>{country}</b> — 
                <span style='background-color:#e8f5e9; color:#2e7d32; padding:4px 10px; border-radius:5px; font-family:monospace;'>{score:.4f}</span></h3>
                <ul style="padding-left:20px; line-height:1.6;">
            """
            for reason in reasons:
                card_html += f"<li>{reason}</li>"
            card_html += "</ul></div>"

            st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("### 📊 Visualization of top scores")
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bar_labels = [code_to_name.get(code, code) for code in result["country_of_destination"]]
        ax.barh(bar_labels, result["final_score"], color="teal", height=0.4)
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

# --- Create unified mapping for both origin and destination countries ---
all_countries_df = pd.concat([
    df[["country_of_birth", "Country_origin"]].rename(columns={"country_of_birth": "code", "Country_origin": "name"}),
    df[["country_of_destination", "Country_dest"]].rename(columns={"country_of_destination": "code", "Country_dest": "name"})
]).drop_duplicates().sort_values("name")

country_display_map = {
    f"{code} - {name}": code
    for code, name in all_countries_df.values
}

# --- Dropdowns for country selection ---
p1_label = st.selectbox("Country 1", list(country_display_map.keys()), key="p1")
p2_label = st.selectbox("Country 2", list(country_display_map.keys()), key="p2")
p1 = country_display_map[p1_label]
p2 = country_display_map[p2_label]

# --- Indicator selection ---
selected_ind = st.multiselect("Select indicators to compare", indicators, default=["Jobs", "Education"])

# --- Helper to retrieve metrics from dest_* or origin_* ---
def get_country_values(code, selected_indicators):
    row = df[df["country_of_destination"] == code]
    prefix = "dest_"
    if row.empty:
        row = df[df["country_of_birth"] == code]
        prefix = "origin_"
    values = row[[f"{prefix}{i}" for i in selected_indicators]].iloc[0]
    values.index = [i.replace(prefix, "") for i in values.index]
    return values

# --- Display chart ---
if selected_ind:
    avg1 = get_country_values(p1, selected_ind)
    avg2 = get_country_values(p2, selected_ind)
    x = range(len(selected_ind))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 0.2 for i in x], avg1.values, width=0.4, label=p1_label)
    ax.bar([i + 0.2 for i in x], avg2.values, width=0.4, label=p2_label)
    ax.set_xticks(x)
    ax.set_xticklabels(selected_ind, rotation=45, ha="right")
    ax.set_ylabel("Normalized Value")
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

# --- Create unified mapping for both origin and destination countries ---
all_countries_df = pd.concat([
    df[["country_of_birth", "Country_origin"]].rename(columns={"country_of_birth": "code", "Country_origin": "name"}),
    df[["country_of_destination", "Country_dest"]].rename(columns={"country_of_destination": "code", "Country_dest": "name"})
]).drop_duplicates().sort_values("name")

# --- Indicator selection ---
selected = st.multiselect(
    "Select at least two indicators to group countries",
    options=indicators,
    default=["Education", "Income"]
)

if len(selected) < 2:
    st.warning("⚠️ Select at least two indexes to continue.")
else:
    try:
        # --- Retrieve values for each country ---
        def get_country_row(code):
            row = df[df["country_of_destination"] == code]
            prefix = "dest_"
            if row.empty:
                row = df[df["country_of_birth"] == code]
                prefix = "origin_"
            values = row[[f"{prefix}{i}" for i in selected]].iloc[0]
            values.index = selected  # for consistency in PCA & plotting
            return values

        data_rows = []
        for _, row in all_countries_df.iterrows():
            code = row["code"]
            name = row["name"]
            vals = get_country_row(code)
            data_rows.append({"code": code, "name": name, **vals})

        df_media = pd.DataFrame(data_rows)

        # === Standardize ===
        X = StandardScaler().fit_transform(df_media[selected])

        # === PCA projection or raw axes ===
        if X.shape[1] > 2:
            X_pca = PCA(n_components=2).fit_transform(X)
            df_media["X"] = X_pca[:, 0]
            df_media["Y"] = X_pca[:, 1]
            x_label, y_label = "Principal Component 1", "Principal Component 2"
        else:
            df_media["X"] = X[:, 0]
            df_media["Y"] = X[:, 1]
            x_label, y_label = selected[0], selected[1]

        # === Optimal number of clusters ===
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

        # === Label clusters ===
        cluster_summary = df_media.groupby("Cluster")[selected].mean()
        cluster_labels = {}
        for cluster_id, row in cluster_summary.iterrows():
            high = [col for col, val in row.items() if val >= 0.7]
            medium = [col for col, val in row.items() if 0.4 <= val < 0.7]
            low = [col for col, val in row.items() if val < 0.4]
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
        df_media["text"] = df_media["code"]

        hover_data = {
            "name": True,
            **{col: True for col in selected}
        }

        # === Dynamic sizing ===
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

st.markdown("### 🌍 Brain Drain or Brain Gain?")
st.info("""
While this tool helps individuals explore better opportunities abroad, it's important to reflect on the broader implications of skilled migration.

Research shows that **brain drain** isn't always a net loss for origin countries. With the right policies, it can actually fuel a **brain gain** effect:

- 🎓 Countries with **flexible education and training systems** can adapt to talent outflows by upskilling their population-triggering a cycle of innovation and resilience.
- 🌐 **Diaspora networks**, return migration, and remittances can boost entrepreneurship, knowledge exchange, and long-term development at home.

This platform supports personal decision-making **while also offering insights** that policymakers could use to anticipate or manage skilled migration trends.

🔗 [Yale EGC – Brain Drain or Brain Gain?](https://egc.yale.edu/brain-drain-or-brain-gain)
""")

st.markdown("### ✔️ In Short:")

st.markdown("""
To sum up the broader message and practical impact of this tool, here are the key takeaways:

- Your insights help **maximize the benefits** of skilled migration by focusing on opportunities—not just destinations.
- Whether you're considering **temporary relocation**, **return plans**, or **career exploration**, this tool empowers informed choices.
- For policymakers, it highlights the importance of investing in education, mobility, and digital infrastructure to **turn brain drain into brain gain**.
""")

st.markdown("""
### 💠 Thank You for Exploring with GoWhere 🌍

We hope this app has given you confidence and clarity. Let the data guide your journey – and may your next destination be one of growth and fulfillment.

If you'd like to explore more variables or update your preferences, just scroll up and try again.
""")



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# === LOAD DATA ===
df = pd.read_csv("dataset_final.csv")

st.set_page_config(page_title="GoWhere - Brain Drain Analyzer", layout="centered")

# === HEADER ===
st.title("🌍 GoWhere - Brain Drain Analyzer")
st.markdown("""
This tool helps you discover the best countries for your personal needs by comparing key indicators
like work, health, environment, and more. Fill in your preferences, get personalized recommendations,
and compare options visually.
""")

# === INDICATORS ===
available_indicators = [
    "Education", "Jobs", "Income", "Safety", "Health", "Environment",
    "Civic engagement", "Accessiblity to services", "Housing",
    "Community", "Life satisfaction", "PR rating", "CL rating"
]

# === USER PREFERENCES ===
st.subheader("👤 Personal Preferences")
sex = st.selectbox("Gender", ["Male", "Female"])
origin = st.selectbox("Country of birth", sorted(df["country_of_birth"].unique()))

# === NEGATIVE FACTORS ===
st.markdown("### ❌ What do you want to improve in your current country?")
st.markdown("Select things you’d like to escape or improve and assign importance (0–10).")
col_neg = st.columns(2)
indices_to_improve = {}
for i, ind in enumerate(available_indicators):
    with col_neg[i % 2]:
        check = st.checkbox(ind, key=f"neg_{ind}")
        if check:
            weight = st.slider(f"Weight for {ind}", 0.0, 10.0, 5.0, 0.5, key=f"w_neg_{ind}")
            indices_to_improve[ind] = weight

# === POSITIVE FACTORS ===
st.markdown("### ✅ What do you desire in a new country?")
st.markdown("Indicate the things you're actively looking for and assign their importance.")
col_pos = st.columns(2)
indices_desired = {}
for i, ind in enumerate(available_indicators):
    with col_pos[i % 2]:
        check = st.checkbox(ind, key=f"pos_{ind}")
        if check:
            weight = st.slider(f"Weight for {ind}", 0.0, 10.0, 5.0, 0.5, key=f"w_pos_{ind}")
            indices_desired[ind] = weight

# === RECOMMENDATION ENGINE ===
if st.button("🔍 Discover best countries"):
    def recommend(df, origin, sex, to_improve, desired, top_n=5):
        df_user = df[(df["country_of_birth"] == origin) & (df["sex"] == sex)].copy()

        def score_row(row):
            score = 0
            reasons = []

            for ind, w in to_improve.items():
                delta = row[f"dest_{ind}"] - row[f"origin_{ind}"]
                score += delta * w
                reasons.append(f"{ind} {'↑' if delta > 0 else '↓'} ({delta:.2f})")

            for ind, w in desired.items():
                val = row[f"dest_{ind}"]
                delta = val - row[f"origin_{ind}"]
                score += val * w
                reasons.append(f"{ind} {'↑' if delta > 0 else '↓'} ({delta:.2f})")

            return pd.Series({"score": score, "reasons": ", ".join(reasons)})

        df_user[["score", "reasons"]] = df_user.apply(score_row, axis=1)

        if df_user["score"].nunique() > 1:
            df_user["score_norm"] = MinMaxScaler().fit_transform(df_user[["score"]])
        else:
            df_user["score_norm"] = 1.0

        if desired:
            profile = np.array(list(desired.values())).reshape(1, -1)
            feature_cols = [f"dest_{k}" for k in desired.keys()]
            mat = df_user[feature_cols].values
            similarity = cosine_similarity(profile, mat)[0]
        else:
            similarity = np.zeros(len(df_user))

        df_user["similarity"] = similarity
        df_user["final_score"] = 0.5 * df_user["score_norm"] + 0.5 * df_user["similarity"]

        ranking = (
            df_user.groupby("country_of_destination")
            .agg({
                "final_score": "mean",
                "reasons": lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0]
            })
            .sort_values("final_score", ascending=False)
            .reset_index()
            .head(top_n)
        )
        return ranking

    results = recommend(df, origin, sex, indices_to_improve, indices_desired)
    st.subheader("🔝 Top Recommended Countries")
    st.dataframe(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(results["country_of_destination"], results["final_score"], color="skyblue")
    ax.set_xlabel("Final Combined Score")
    ax.set_title("Top Recommendations")
    ax.invert_yaxis()
    st.pyplot(fig)

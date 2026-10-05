"""
Streamlit app using the saved Random Forest pipeline.

1) python train_model.py          (creates Voting_pipeline.pkl)
2) streamlit run app_rf.py
"""
import joblib
import pandas as pd
import streamlit as st

CSV_PATH = "National.csv"
BUNDLE_PATH = "Voting_pipeline.pkl"
ELECTION_DATE = "2026-11-04"
DEFAULT_MUNI = "ETH - eThekwini"

st.set_page_config(page_title="eThekwini 2026 - Random Forest", page_icon="🗳️", layout="wide")


@st.cache_resource
def load_bundle():
    return joblib.load(BUNDLE_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH)


def ward_sort(w):
    return (len(str(w)), str(w))


try:
    bundle = load_bundle()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"{e.filename} not found. Run `python train_model.py` first and keep National.csv in this folder.")
    st.stop()

model, le, FEATURES = bundle["model"], bundle["label_encoder"], bundle["features"]


def predict(X):
    proba = model.predict_proba(X[FEATURES])
    names = le.inverse_transform(model.named_steps["random_forest"].classes_.astype(int))
    return pd.DataFrame(proba, columns=names, index=X.index)


st.title("🗳️ eThekwini 2026 - Random Forest forecast")
st.caption("Estimates from historical patterns - not confirmed election results.")

m = bundle["metrics"]
c1, c2, c3 = st.columns(3)
c1.metric("Train accuracy", f"{100 * m['train_acc']:.1f}%")
c2.metric("Test accuracy", f"{100 * m['test_acc']:.1f}%")
c3.metric("5-fold CV", f"{100 * m['cv_mean']:.1f}% ± {100 * m['cv_std']:.1f}")

munis = sorted(df["Municipality"].dropna().unique())
muni = st.sidebar.selectbox("Municipality", munis,
                            index=munis.index(DEFAULT_MUNI) if DEFAULT_MUNI in munis else 0)
mdf = df[df["Municipality"] == muni]

tab1, tab2 = st.tabs(["Ward forecast", "Single station"])

# ---- Ward forecast ---------------------------------------------------------
with tab1:
    wards = sorted(mdf["Ward"].dropna().unique(), key=ward_sort)
    chosen = st.multiselect("Target wards (empty = all)", wards)
    target = mdf[mdf["Ward"].isin(chosen)] if chosen else mdf

    if st.button("Run forecast", type="primary"):
        base = target[FEATURES].drop_duplicates().copy()
        base["DateGenerated"] = ELECTION_DATE
        proba = predict(base)
        ward_p = proba.groupby(base["Ward"].astype(str)).mean()
        out = pd.DataFrame({
            "Predicted winner": ward_p.idxmax(axis=1),
            "Confidence %": (100 * ward_p.max(axis=1)).round(1),
        }).sort_index(key=lambda s: s.map(lambda w: (len(w), w)))

        st.bar_chart(out["Predicted winner"].value_counts())
        st.dataframe(out.rename_axis("Ward"))
        st.download_button("Download CSV", out.rename_axis("Ward").to_csv().encode(),
                           "forecast_2026.csv", "text/csv")

# ---- Single station --------------------------------------------------------
with tab2:
    ward = st.selectbox("Ward", sorted(mdf["Ward"].dropna().unique(), key=ward_sort), key="w")
    wdf = mdf[mdf["Ward"] == ward]
    district = st.selectbox("Voting district", sorted(wdf["VotingDistrict"].unique()))
    ddf = wdf[wdf["VotingDistrict"] == district]
    station = st.selectbox("Voting station", sorted(ddf["VotingStationName"].dropna().unique()))
    sdf = ddf[ddf["VotingStationName"] == station]
    ballot = st.selectbox("Ballot type", sorted(sdf["BallotType"].unique()))
    r = sdf[sdf["BallotType"] == ballot].iloc[0]

    a, b = st.columns(2)
    reg = a.number_input("Registered voters", 0, value=int(r["RegisteredVoters"]), step=10)
    spoilt = b.number_input("Spoilt votes", 0, value=int(r["SpoiltVotes"]), step=1)

    if st.button("Predict station"):
        X = pd.DataFrame([{**r[FEATURES].to_dict(), "VotingDistrict": district,
                           "VotingStationName": station, "BallotType": ballot,
                           "RegisteredVoters": reg, "SpoiltVotes": spoilt,
                           "DateGenerated": ELECTION_DATE}])
        p = predict(X).iloc[0].sort_values(ascending=False)
        st.success(f"Predicted winner: **{p.index[0]}** ({100 * p.iloc[0]:.1f}%)")
        st.bar_chart((100 * p).rename("Probability %"))
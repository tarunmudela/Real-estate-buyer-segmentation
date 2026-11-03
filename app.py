"""Streamlit dashboard. Run ml_pipeline.py first, then:  streamlit run app.py"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
SEGMENT_NAMES = {0: "Dissatisfied Mainstream", 1: "Satisfied Mainstream",
                 2: "Portfolio Accumulators", 3: "Premium-Ticket Buyers"}

st.set_page_config(page_title="Parcl Buyer Segmentation", layout="wide")


@st.cache_data
def get_data():
    return pd.read_csv(os.path.join(HERE, "clustered_clients.csv"))


if not os.path.exists(os.path.join(HERE, "clustered_clients.csv")):
    st.error("clustered_clients.csv not found. Run:  python ml_pipeline.py")
    st.stop()

df = get_data()
order = list(SEGMENT_NAMES.values())

st.sidebar.header("Filters")
sub = df
countries = st.sidebar.multiselect("Country", sorted(df.country.unique()))
sub = sub[sub.country.isin(countries)] if countries else sub
regions = st.sidebar.multiselect("Region", sorted(sub.region.unique()))
sub = sub[sub.region.isin(regions)] if regions else sub
purposes = st.sidebar.multiselect("Acquisition purpose", sorted(df.acquisition_purpose.unique()))
sub = sub[sub.acquisition_purpose.isin(purposes)] if purposes else sub
types = st.sidebar.multiselect("Client type", sorted(df.client_type.unique()))
sub = sub[sub.client_type.isin(types)] if types else sub

st.title("Buyer Segmentation & Investment Profiling")
st.caption(f"{len(sub):,} of {len(df):,} clients match the current filters")
if sub.empty:
    st.warning("No clients match these filters.")
    st.stop()

t1, t2, t3, t4 = st.tabs(["Segmentation Overview", "Investor Behavior",
                          "Geographic Analysis", "Segment Insights"])
with t1:
    a, b = st.columns(2)
    counts = sub.segment.value_counts().reindex(order).fillna(0).reset_index()
    counts.columns = ["segment", "clients"]
    a.plotly_chart(px.bar(counts, x="segment", y="clients", color="segment",
                          title="Clients per segment"), use_container_width=True)
    b.plotly_chart(px.scatter(sub, x="pc1", y="pc2", color="segment", opacity=0.6,
                              title="Segments in PCA space", category_orders={"segment": order}),
                   use_container_width=True)
    m = st.columns(4)
    m[0].metric("Clients", f"{len(sub):,}")
    m[1].metric("Avg satisfaction", f"{sub.satisfaction_score.mean():.2f}")
    m[2].metric("Avg units bought", f"{sub.units_bought.mean():.2f}")
    m[3].metric("Avg total spend", f"${sub.total_spend.mean():,.0f}")

with t2:
    g = sub.groupby("segment").agg(
        investment_rate=("acquisition_purpose", lambda s: (s == "Investment").mean()),
        loan_rate=("loan_applied", lambda s: (s == "Yes").mean()),
        avg_spend=("total_spend", "mean")).reindex(order).dropna(how="all").reset_index()
    a, b = st.columns(2)
    a.plotly_chart(px.bar(g, x="segment", y=["investment_rate", "loan_rate"], barmode="group",
                          title="Investment and loan rates by segment"), use_container_width=True)
    b.plotly_chart(px.bar(g, x="segment", y="avg_spend", color="segment",
                          title="Average total spend"), use_container_width=True)
    c, d = st.columns(2)
    c.plotly_chart(px.box(sub, x="segment", y="avg_price", color="segment",
                          title="Average price per unit", category_orders={"segment": order}),
                   use_container_width=True)
    ref = pd.crosstab(sub.segment, sub.referral_channel, normalize="index").reset_index().melt(
        "segment", var_name="channel", value_name="share")
    d.plotly_chart(px.bar(ref, x="segment", y="share", color="channel",
                          title="Referral channel mix"), use_container_width=True)

with t3:
    geo = sub.groupby(["country", "segment"]).size().reset_index(name="clients")
    st.plotly_chart(px.bar(geo, x="country", y="clients", color="segment", title="Segments by country",
                           category_orders={"segment": order}), use_container_width=True)
    top = sub.region.value_counts().head(15).index
    reg = sub[sub.region.isin(top)].groupby(["region", "segment"]).size().reset_index(name="clients")
    st.plotly_chart(px.bar(reg, x="region", y="clients", color="segment",
                           title="Segments in the 15 largest regions"), use_container_width=True)
    stats = sub.groupby("country").agg(
        clients=("client_id", "count"),
        investment_rate=("acquisition_purpose", lambda s: (s == "Investment").mean()),
        loan_rate=("loan_applied", lambda s: (s == "Yes").mean()),
        avg_spend=("total_spend", "mean")).round(3)
    st.dataframe(stats.sort_values("clients", ascending=False), use_container_width=True)

with t4:
    choice = st.selectbox("Segment", order)
    s = sub[sub.segment == choice]
    if s.empty:
        st.info("No clients from this segment match the filters.")
    else:
        st.subheader(f"{choice} ({len(s):,} clients)")
        cols = ["age", "satisfaction_score", "units_bought", "total_spend", "avg_price", "avg_area", "office_share"]
        st.dataframe(s[cols].describe().T.round(2), use_container_width=True)
        x, y, z = st.columns(3)
        x.metric("Investment share", f"{(s.acquisition_purpose == 'Investment').mean():.1%}")
        y.metric("Loan share", f"{(s.loan_applied == 'Yes').mean():.1%}")
        z.metric("Company share", f"{(s.client_type == 'Company').mean():.1%}")
        st.download_button("Download this segment (CSV)", s.to_csv(index=False), f"{choice}.csv", "text/csv")

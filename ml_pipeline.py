"""
ML pipeline: buyer segmentation for real estate market intelligence.
Cleans clients.csv and properties.csv, encodes and scales features, picks k with
elbow + silhouette, fits K-Means (validated with hierarchical clustering) and saves
clustered_clients.csv, cluster_profiles.csv and model_selection.csv.

Usage:  python ml_pipeline.py [data_folder]     (default: the folder of this file)
Requirements: pandas numpy scikit-learn
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = next((a for a in sys.argv[1:] if os.path.isdir(a)), HERE)
REF_YEAR = 2024          # transactions are dated 2024
K = 4                    # number of segments (silhouette alone prefers 3; see paper)
SEGMENT_NAMES = {0: "Dissatisfied Mainstream", 1: "Satisfied Mainstream",
                 2: "Portfolio Accumulators", 3: "Premium-Ticket Buyers"}


# --------------------------------------------------------------------------
# Step 1 - data cleaning
# --------------------------------------------------------------------------
def load_and_clean(data_dir=DATA_DIR):
    c = pd.read_csv(os.path.join(data_dir, "clients.csv"))
    p = pd.read_csv(os.path.join(data_dir, "properties.csv"))

    for col in ["client_type", "gender", "country", "region",
                "acquisition_purpose", "loan_applied", "referral_channel"]:
        c[col] = c[col].astype(str).str.strip()
    c["client_type"] = c["client_type"].str.title().replace({"Corporate": "Company"})
    c["acquisition_purpose"] = c["acquisition_purpose"].str.title()
    c["loan_applied"] = c["loan_applied"].str.title()
    c["country"] = c["country"].replace({"United States": "USA", "US": "USA", "United Kingdom": "UK"})
    c = c.drop_duplicates("client_id")

    # date_of_birth has mixed formats (05-11-1968, 11/26/1962): the year is enough for age
    c["birth_year"] = pd.to_numeric(c["date_of_birth"].str.extract(r"(\d{4})")[0], errors="coerce")
    c["age"] = (REF_YEAR - c["birth_year"]).where(lambda s: s.between(18, 100))
    c["age"] = c["age"].fillna(c["age"].median())
    c["satisfaction_score"] = c["satisfaction_score"].fillna(c["satisfaction_score"].median())

    p["price"] = pd.to_numeric(p["sale_price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    p["unit_category"] = p["unit_category"].str.strip().str.title()
    sold = p[(p["listing_status"] == "Sold") & p["client_ref"].notna()].copy()
    sold["is_office"] = (sold["unit_category"] == "Office").astype(int)

    agg = (sold.groupby("client_ref")
           .agg(units_bought=("listing_id", "count"), total_spend=("price", "sum"),
                avg_price=("price", "mean"), avg_area=("floor_area_sqft", "mean"),
                office_share=("is_office", "mean"))
           .reset_index().rename(columns={"client_ref": "client_id"}))
    df = c.merge(agg, on="client_id", how="left")
    for col in ["units_bought", "total_spend"]:
        df[col] = df[col].fillna(0)
    for col in ["avg_price", "avg_area", "office_share"]:
        df[col] = df[col].fillna(df[col].median())
    df["log_total_spend"] = np.log1p(df["total_spend"])
    return df


# --------------------------------------------------------------------------
# Steps 2 and 3 - encoding and scaling
# --------------------------------------------------------------------------
def build_features(df):
    flags = pd.DataFrame({
        "loan_flag": (df["loan_applied"] == "Yes").astype(int),
        "invest_flag": (df["acquisition_purpose"] == "Investment").astype(int),
        "company_flag": (df["client_type"] == "Company").astype(int),
        "intl_flag": (df["country"] != "USA").astype(int)})
    onehot = pd.get_dummies(df[["referral_channel", "gender"]]).astype(int)
    num_cols = ["age", "satisfaction_score", "units_bought", "log_total_spend",
                "avg_price", "avg_area", "office_share"]
    scaled = pd.DataFrame(StandardScaler().fit_transform(df[num_cols]), columns=num_cols, index=df.index)
    return pd.concat([scaled, flags, onehot], axis=1).values


# --------------------------------------------------------------------------
# Steps 4 and 5 - model selection (elbow + silhouette) and Step 6 - clustering
# --------------------------------------------------------------------------
def model_selection(X, k_range=range(2, 11)):
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        hc = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(X)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette_kmeans": silhouette_score(X, km.labels_),
                     "silhouette_hierarchical": silhouette_score(X, hc.labels_),
                     "ari_kmeans_vs_hierarchical": adjusted_rand_score(km.labels_, hc.labels_)})
    return pd.DataFrame(rows)


def run_analysis(data_dir=DATA_DIR, out_dir=HERE, select_k=True):
    df = load_and_clean(data_dir)
    X = build_features(df)
    selection = model_selection(X) if select_k else None

    km = KMeans(n_clusters=K, n_init=20, random_state=42).fit(X)
    hc = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(X)
    df["cluster"], df["hc_cluster"] = km.labels_, hc.labels_
    df["segment"] = df["cluster"].map(SEGMENT_NAMES)
    pcs = PCA(2, random_state=42).fit_transform(X)
    df["pc1"], df["pc2"] = pcs[:, 0], pcs[:, 1]

    profile = df.groupby("segment").agg(
        clients=("client_id", "count"), avg_age=("age", "mean"), satisfaction=("satisfaction_score", "mean"),
        investment_rate=("acquisition_purpose", lambda s: (s == "Investment").mean()),
        loan_rate=("loan_applied", lambda s: (s == "Yes").mean()),
        company_rate=("client_type", lambda s: (s == "Company").mean()),
        avg_units=("units_bought", "mean"), avg_spend=("total_spend", "mean"),
        avg_price=("avg_price", "mean"), office_share=("office_share", "mean")).round(3)

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "clustered_clients.csv"), index=False)
    profile.to_csv(os.path.join(out_dir, "cluster_profiles.csv"))
    if selection is not None:
        selection.to_csv(os.path.join(out_dir, "model_selection.csv"), index=False)
    return df, profile, selection


if __name__ == "__main__":
    df, profile, selection = run_analysis()
    print(selection.round(4).to_string(index=False))
    print()
    print(profile.to_string())
    print(f"\nSaved clustered_clients.csv, cluster_profiles.csv, model_selection.csv in {HERE}")

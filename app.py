# =====================================
# STREAMLIT AIR QUALITY DASHBOARD APP
# (DEPLOYMENT-READY)
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# -------------------------------------
# CONFIG
# -------------------------------------
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]
SEASON_ORDER = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]


# -------------------------------------
# LOAD + ENRICH DATA
# -------------------------------------
@st.cache_data
def load_data_from_path(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return enrich_df(df)

@st.cache_data
def load_data_from_upload(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    return enrich_df(df)

def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Ensure numeric
    if "Month" in df.columns:
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    if "AQI" in df.columns:
        df["AQI"] = pd.to_numeric(df["AQI"], errors="coerce")

    # Create MonthName if missing
    if "Month" in df.columns and "MonthName" not in df.columns:
        month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        df["MonthName"] = df["Month"].map(month_map)

    # Create Season if missing
    if "Month" in df.columns and "Season" not in df.columns:
        def month_to_season(m):
            if m in [12, 1, 2]:
                return "Winter"
            elif m in [3, 4, 5]:
                return "Pre-monsoon"
            elif m in [6, 7, 8, 9]:
                return "Monsoon"
            elif m in [10, 11]:
                return "Post-monsoon"
            else:
                return np.nan
        df["Season"] = df["Month"].map(month_to_season)

    return df

# -------------------------------------
# SIDEBAR FILTERS + DOWNLOAD
# -------------------------------------
def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    out = df.copy()

    if "City" in out.columns:
        city = st.sidebar.selectbox("City", ["All"] + sorted(out["City"].dropna().unique().tolist()))
        if city != "All":
            out = out[out["City"] == city]

    if "Season" in out.columns:
        season = st.sidebar.selectbox("Season", ["All"] + sorted(out["Season"].dropna().unique().tolist()))
        if season != "All":
            out = out[out["Season"] == season]

    if "RegionType" in out.columns:
        region = st.sidebar.selectbox("Region Type", ["All"] + sorted(out["RegionType"].dropna().unique().tolist()))
        if region != "All":
            out = out[out["RegionType"] == region]

    if "Year" in out.columns and out["Year"].notna().any():
        years = sorted(out["Year"].dropna().unique())
        if years:
            year_min, year_max = int(min(years)), int(max(years))
            year_range = st.sidebar.slider("Year Range", year_min, year_max, (year_min, year_max))
            out = out[(out["Year"] >= year_range[0]) & (out["Year"] <= year_range[1])]

    # Download filtered CSV
    csv = out.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button("Download filtered data (CSV)", csv, "filtered_data.csv", "text/csv")

    return out

# -------------------------------------
# PAGES
# -------------------------------------
def page_preprocessing(df: pd.DataFrame):
    st.title("Data Preprocessing")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Cities", df["City"].nunique() if "City" in df.columns else 0)
    c3.metric("Mean AQI", f"{df['AQI'].mean():.1f}" if "AQI" in df.columns else "N/A")

    st.subheader("Dataset preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Missing values")
    st.dataframe(df.isna().sum().to_frame("Missing count"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        if "AQI" in df.columns:
            fig = px.histogram(df, x="AQI", nbins=40, title="AQI Distribution")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if "AQI_Bucket_Label" in df.columns:
            vc = df["AQI_Bucket_Label"].value_counts().reset_index()
            vc.columns = ["AQI_Bucket_Label", "count"]
            fig = px.bar(vc, x="AQI_Bucket_Label", y="count", title="AQI Category Distribution")
            st.plotly_chart(fig, use_container_width=True)

def page_trends(df: pd.DataFrame):
    st.title("Identifying Trends")

    if {"Year", "Month", "AQI"}.issubset(df.columns):
        df2 = df.dropna(subset=["Year", "Month", "AQI"]).copy()
        df2["YearMonth"] = pd.to_datetime(
            df2["Year"].astype(int).astype(str) + "-" + df2["Month"].astype(int).astype(str) + "-01",
            errors="coerce"
        )
        df2 = df2.dropna(subset=["YearMonth"])

        monthly = df2.groupby("YearMonth", as_index=False)["AQI"].mean()
        fig = px.line(monthly, x="YearMonth", y="AQI", title="Monthly Mean AQI (filtered)")
        st.plotly_chart(fig, use_container_width=True)

        if "City" in df2.columns:
            multi = df2.groupby(["City", "YearMonth"], as_index=False)["AQI"].mean()
            fig = px.line(multi, x="YearMonth", y="AQI", color="City", title="Monthly Mean AQI by City (filtered)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need Year, Month and AQI columns to plot trends.")

    if "Season" in df.columns and "AQI" in df.columns:
        season_aqi = df.groupby("Season", as_index=False)["AQI"].mean()
        season_aqi["Season"] = pd.Categorical(season_aqi["Season"], categories=SEASON_ORDER, ordered=True)
        season_aqi = season_aqi.sort_values("Season")
        fig = px.bar(season_aqi, x="Season", y="AQI", title="Mean AQI by Season")
        st.plotly_chart(fig, use_container_width=True)

def page_relationships(df: pd.DataFrame):
    st.title("Relationships Between Pollutants")

    if "AQI" not in df.columns:
        st.warning("AQI column not found.")
        return

    available = [c for c in POLLUTANT_COLS if c in df.columns]
    if not available:
        st.warning("No pollutant columns found.")
        return

    corr = df[available + ["AQI"]].corr(numeric_only=True)
    fig = px.imshow(corr, title="Correlation Heatmap", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

    pollutant = st.selectbox("Select pollutant", available)
    fig = px.scatter(df, x=pollutant, y="AQI", trendline="ols", title=f"AQI vs {pollutant}")
    st.plotly_chart(fig, use_container_width=True)

    if "Season" in df.columns:
        season_pollutants = df.groupby("Season")[available].mean().reindex(SEASON_ORDER)
        fig = px.imshow(
            season_pollutants,
            labels=dict(x="Pollutant", y="Season", color="Mean Level"),
            title="Seasonal Mean Pollutant Levels",
            aspect="auto"
        )
        fig.update_xaxes(side="top")
        st.plotly_chart(fig, use_container_width=True)

def page_geography(df: pd.DataFrame):
    st.title("Geographic Variation")

    if "City" in df.columns and "AQI" in df.columns:
        city_aqi = df.groupby("City", as_index=False)["AQI"].mean().sort_values("AQI", ascending=False)
        fig = px.bar(city_aqi, x="City", y="AQI", title="Mean AQI by City")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need City and AQI columns for city comparison.")

    if "RegionType" in df.columns and "AQI" in df.columns:
        region_aqi = df.groupby("RegionType", as_index=False)["AQI"].mean().sort_values("AQI", ascending=False)
        fig = px.bar(region_aqi, x="RegionType", y="AQI", title="Mean AQI by Region Type")
        st.plotly_chart(fig, use_container_width=True)

    if "AQI_Bucket_Label" in df.columns and "City" in df.columns:
        buckets = df.groupby(["City", "AQI_Bucket_Label"]).size().reset_index(name="count")
        fig = px.bar(buckets, x="City", y="count", color="AQI_Bucket_Label", title="AQI Buckets by City")
        st.plotly_chart(fig, use_container_width=True)

def page_ml(df: pd.DataFrame):
    st.title("Machine Learning")

    if "AQI" not in df.columns:
        st.warning("No AQI column to predict.")
        return

    features = [c for c in POLLUTANT_COLS if c in df.columns]
    if not features:
        st.warning("No pollutant features available to train a model.")
        return

    data = df[features + ["AQI"]].dropna()
    if len(data) < 50:
        st.warning(f"Not enough rows after dropping NaNs ({len(data)} rows).")
        return

    test_size = st.sidebar.slider("ML test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.sidebar.number_input("ML random seed", value=42, step=1)

    X = data[features]
    y = data["AQI"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state)
    )

    n_estimators = st.slider("n_estimators", 100, 1000, 300, 50)
    max_depth = st.slider("max_depth", 2, 30, 12, 1)

    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        random_state=int(random_state),
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    c1, c2 = st.columns(2)
    c1.metric("MAE", f"{mean_absolute_error(y_test, preds):.2f}")
    c2.metric("R²", f"{r2_score(y_test, preds):.3f}")

    pv = pd.DataFrame({"Actual": y_test.values, "Predicted": preds})
    fig = px.scatter(pv, x="Actual", y="Predicted", title="Predicted vs Actual AQI")
    st.plotly_chart(fig, use_container_width=True)

    imp = pd.DataFrame({"Feature": features, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False)
    fig = px.bar(imp, x="Feature", y="Importance", title="Feature Importance")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------
# MAIN
# -------------------------------------
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Data Preprocessing", "Identifying Trends", "Relationships", "Geographic Variation", "Machine Learning"]
    )

    st.sidebar.subheader("Data Source")
    mode = st.sidebar.radio("Choose input", ["Upload CSV (recommended)", "Load from path"], index=0)

    if mode == "Upload CSV (recommended)":
        uploaded = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
        if uploaded is None:
            st.title("Air Quality Dashboard")
            st.info("Upload your CSV using the sidebar to begin.")
            st.stop()
        df = load_data_from_upload(uploaded)
    else:
        path = st.sidebar.text_input("CSV path", value=DEFAULT_LOCAL_PATH)
        if not path:
            st.title("Air Quality Dashboard")
            st.info("Enter a valid CSV path in the sidebar, or switch to Upload mode.")
            st.stop()
        df = load_data_from_path(path)

    df_filtered = apply_filters(df)

    if page == "Data Preprocessing":
        page_preprocessing(df_filtered)
    elif page == "Identifying Trends":
        page_trends(df_filtered)
    elif page == "Relationships":
        page_relationships(df_filtered)
    elif page == "Geographic Variation":
        page_geography(df_filtered)
    else:
        page_ml(df_filtered)

if __name__ == "__main__":
    main()

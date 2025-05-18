# real_estate_predictor_app.py

import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import gdown

# === CONFIG ===
PARQUET_DRIVE_ID = "17tWMJLlBIpxirNTi2JntwJZ0Pt7lMgC9"
PKL_DRIVE_ID = "1nEgmP6L8OgHN1Zhx5rdXEPFyIpgsllvX"

@st.cache_data
def load_data():
    url = f"https://drive.google.com/uc?id={PARQUET_DRIVE_ID}"
    output = "cleaned_real_estate_ready.parquet"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return pd.read_parquet(output)

@st.cache_resource
def load_model():
    import joblib
    import sklearn.ensemble._forest  # Fix for RandomForestRegressor unpickling
    url = f"https://drive.google.com/uc?id={PKL_DRIVE_ID}"
    output = "price_predictor_model_v3.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    return joblib.load(output)

# === LOAD ===
df = load_data()
model = load_model()

st.set_page_config(page_title="Dubai Real Estate Price Predictor", layout="wide")
st.title("Dubai Property Price Predictor")
st.markdown("Predict residential property prices using real transaction data.")

# === MODE SELECTION ===
mode = st.radio("Choose mode:", ["🔍 Browse Listings", "✍️ Predict a Property"], horizontal=True)

if mode == "🔍 Browse Listings":
    st.subheader("Filter and Predict Prices from Database")

    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.selectbox("Select Area", sorted(df['area_name_en'].dropna().unique()))
    with col2:
        prop_type = st.selectbox("Select Property Type", sorted(df['property_type_en'].dropna().unique()))
    with col3:
    if 'developer_en' in df.columns:
        developer = st.selectbox("Select Developer (optional)", ["All"] + sorted(df['developer_en'].dropna().unique().tolist()))
    else:
        developer = "All"

    df_filtered = df[
        (df['area_name_en'] == area) &
        (df['property_type_en'] == prop_type)
    ]
    if developer != "All":
        df_filtered = df_filtered[df_filtered['developer_en'] == developer]

    if df_filtered.empty:
        st.warning("No properties match your filters.")
    else:
        df_filtered = df_filtered.copy()
        df_filtered['rooms_en'] = df_filtered['rooms_en'].astype(str).str.extract(r'(\d+)').astype(float)

        input_cols = [
            'procedure_area', 'year', 'years_since_handover', 'rooms_en', 'has_parking',
            'meter_sale_price', 'area_name_en', 'property_type_en', 'property_sub_type_en', 'property_usage_en'
        ]
        df_model = df_filtered[input_cols].dropna()
        df_model = pd.get_dummies(df_model, columns=[
            'area_name_en', 'property_type_en', 'property_sub_type_en', 'property_usage_en'
        ], drop_first=True)

        for col in model.feature_names_in_:
            if col not in df_model.columns:
                df_model[col] = 0
        df_model = df_model[model.feature_names_in_]

        df_filtered = df_filtered.loc[df_model.index].copy()
        df_filtered['Predicted Price (AED)'] = model.predict(df_model).round(0)

        st.dataframe(df_filtered[['transaction_id', 'project_name_en', 'procedure_area', 'rooms_en', 'meter_sale_price', 'Predicted Price (AED)']].head(1000))

        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Results", csv, file_name="filtered_predictions.csv")

# === ✍️ Predict a Property Mode ===

elif mode == "✍️ Predict a Property":
    with st.form("manual_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            procedure_area = st.number_input("Area (sqm)", min_value=10.0, value=100.0)
            rooms_en = st.number_input("Number of Rooms", min_value=0.0, value=2.0)
            has_parking = st.selectbox("Has Parking?", [0, 1])
            year = st.number_input("Transaction Year", min_value=2000, max_value=2025, value=2023)
        with col2:
            area_name = st.selectbox("Area", sorted(df['area_name_en'].dropna().unique()))
            property_type = st.selectbox("Property Type", sorted(df['property_type_en'].dropna().unique()))
            property_sub_type = st.selectbox("Property Sub-Type", sorted(df['property_sub_type_en'].dropna().unique()))
            property_usage = st.selectbox("Property Usage", sorted(df['property_usage_en'].dropna().unique()))

        meter_sale_price = st.number_input("Meter Sale Price (AED/sqm)", min_value=1000.0, value=12000.0)
        wait_years = st.slider("Years to wait (for projected appreciation)", 1, 20, 5)

        submitted = st.form_submit_button("Predict Price")

        if submitted:
            years_since_handover = 2023 - year
            row = pd.DataFrame([{
                'procedure_area': procedure_area,
                'year': year,
                'years_since_handover': years_since_handover,
                'rooms_en': rooms_en,
                'has_parking': has_parking,
                'meter_sale_price': meter_sale_price,
                'area_name_en': area_name,
                'property_type_en': property_type,
                'property_sub_type_en': property_sub_type,
                'property_usage_en': property_usage
            }])

            row_encoded = pd.get_dummies(row, columns=[
                'area_name_en', 'property_type_en', 'property_sub_type_en', 'property_usage_en'
            ], drop_first=True)

            for col in model.feature_names_in_:
                if col not in row_encoded.columns:
                    row_encoded[col] = 0
            row_encoded = row_encoded[model.feature_names_in_]

            pred_price = model.predict(row_encoded)[0]
            st.success(f"Estimated Property Price: AED {pred_price:,.0f}")

            # Market Comparison
            avg_df = df[
                (df['area_name_en'] == area_name) &
                (df['property_type_en'] == property_type)
            ].copy()
            if not avg_df.empty:
                avg_df['rooms_en'] = avg_df['rooms_en'].astype(str).str.extract(r'(\d+)').astype(float)
                avg_df = avg_df.dropna(subset=['procedure_area', 'actual_worth'])
                avg_df['price_per_sqm'] = avg_df['actual_worth'] / avg_df['procedure_area']
                avg_price_per_sqm = avg_df['price_per_sqm'].mean()

                your_total = procedure_area * meter_sale_price
                area_average_total = procedure_area * avg_price_per_sqm
                diff_pct = (your_total - area_average_total) / area_average_total * 100

                if diff_pct > 0:
                    st.info(f"Your input price is **{diff_pct:.1f}% higher** than the area average.")
                elif diff_pct < 0:
                    st.info(f"Your input price is **{abs(diff_pct):.1f}% lower** than the area average.")
                else:
                    st.info("Your input price is **exactly at** the area average.")

                # Growth stage trend
                stage_df = df[
                    (df['area_name_en'] == area_name) &
                    (df['property_type_en'] == property_type) &
                    (df['price_per_sqm'].notna())
                ]
                if not stage_df.empty and 'growth_stage' in stage_df.columns:
                    trend = stage_df.groupby('growth_stage')['price_per_sqm'].mean().reindex([
                        'Off-Plan or Launch', 'Early Growth', 'Maturity', 'Stabilized / Legacy'
                    ])
                    st.subheader("📈 Growth Stage Pricing Trend")
                    st.bar_chart(trend)

                    last_stage = 'Early Growth' if years_since_handover <= 3 else ('Maturity' if years_since_handover <= 7 else 'Stabilized / Legacy')
                    current_price = trend.get(last_stage, avg_price_per_sqm)
                    potential_stage = 'Maturity' if last_stage == 'Early Growth' else 'Stabilized / Legacy'
                    future_price = trend.get(potential_stage, current_price)
                    appreciation = ((future_price - current_price) / current_price) * 100
                    projected_price = your_total * (1 + appreciation / 100)

                    st.info(f"If the property reaches '{potential_stage}' stage in {wait_years} years, projected value could be AED {projected_price:,.0f} (approx. {appreciation:.1f}% increase).")

                    # Area comparison
                    st.subheader("🏙️ Compare with Similar Units in Other Areas")
                    similar_units = df[
                        (df['property_type_en'] == property_type) &
                        (df['property_sub_type_en'] == property_sub_type) &
                        (df['rooms_en'].astype(str).str.extract(r'(\d+)').astype(float).fillna(0).astype(int) == int(rooms_en)) &
                        (df['procedure_area'].between(procedure_area * 0.9, procedure_area * 1.1))
                    ]
                    area_comparison = similar_units.groupby('area_name_en').agg(
                        avg_price_per_sqm=('meter_sale_price', 'mean'),
                        count=('transaction_id', 'count')
                    ).sort_values('avg_price_per_sqm', ascending=False).head(10)
                    st.dataframe(area_comparison.reset_index())

                    # Developer comparison
                    if 'developer_en' in df.columns:
                        st.subheader("🏗️ Compare with Other Developers in Same Area")
                        dev_comparison = df[
                            (df['area_name_en'] == area_name) &
                            (df['property_type_en'] == property_type)
                        ].groupby('developer_en').agg(
                            avg_price_per_sqm=('meter_sale_price', 'mean'),
                            count=('transaction_id', 'count')
                        ).sort_values('avg_price_per_sqm', ascending=False).head(10)
                        st.dataframe(dev_comparison.reset_index())
                        best_dev = dev_comparison['avg_price_per_sqm'].idxmax()
                        st.success(f"🏗️ Top Developer in {area_name}: {best_dev}")
                    else:
                        st.info("Developer data not available in this dataset.")

                    best_area = area_comparison['avg_price_per_sqm'].idxmax()
                    st.success(f"📌 Top Value Area for Similar Units: {best_area}")

                    if 'developer_en' in df.columns and 'dev_comparison' in locals():
                        best_dev = dev_comparison['avg_price_per_sqm'].idxmax()
                        st.success(f"🏗️ Top Developer in {area_name}: {best_dev}")
                    # Recommendation Summary (Styled)
                    with st.expander("📋 Open Recommendation Summary"):
                        st.markdown(
                            f"""
                            <div style='background-color:#f0f9ff;padding:1.5rem;border-radius:10px;'>
                                <h4 style='color:#036;'>🧭 Summary for Your Property</h4>
                                <p>Based on your selection in <strong>{area_name}</strong>, the predicted property value is <strong style='color:#076'>{pred_price:,.0f} AED</strong>.</p>
                                <p>If you wait <strong>{wait_years} years</strong>, your property could appreciate by <strong>{appreciation:.1f}%</strong>, potentially reaching <strong style='color:#076'>{projected_price:,.0f} AED</strong> assuming it matures into the '<strong>{potential_stage}</strong>' stage.</p>
                                <p>Meanwhile, similar units in <strong>{best_area}</strong> offer the highest current average price per sqm, and <strong>{best_dev}</strong> is the leading developer in your selected area.</p>
                                <hr>
                                <ul>
                                    <li>✅ Retain this property if the area is expected to grow.</li>
                                    <li>🔁 Explore relocation to <strong>{best_area}</strong> for higher resale potential.</li>
                                    <li>🏗️ Consider buying from <strong>{best_dev}</strong> for stronger value performance.</li>
                                </ul>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )



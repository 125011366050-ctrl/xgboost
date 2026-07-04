import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.set_page_config(
    page_title="CDSS - Diabetes Decision Support System",
    layout="wide"
)

st.title("🩺 CDSS - Diabetes Decision Support System (XGBoost)")

# Load engine once
if "engine" not in st.session_state:
    from engine import GlucoseEngine
    st.session_state.engine = GlucoseEngine()

engine = st.session_state.engine

st.subheader("📥 Enter Last 10 CGM Readings (5-min intervals)")

# Initialize Input Table
if "cgm_data" not in st.session_state:
    st.session_state.cgm_data = pd.DataFrame({
        "Time": [
            (datetime.now() - timedelta(minutes=5 * i)).strftime("%H:%M")
            for i in range(9, -1, -1)
        ],
        "Glucose": [120] * 10,
        "HR": [75] * 10,
        "Carbs": [0] * 10,
        "Protein": [0] * 10,
        "Fat": [0] * 10,
        "Fiber": [0] * 10,
        "Calories": [0] * 10,
        "Meal_Flag": [0] * 10,
    })

# Input Form
with st.form("cdss_form"):
    entries = st.data_editor(
        st.session_state.cgm_data,
        key="cgm_editor",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "Glucose": st.column_config.NumberColumn(
                "Glucose (mg/dL)",
                min_value=40,
                max_value=400,
                step=1
            ),
            "HR": st.column_config.NumberColumn(
                "Heart Rate (bpm)",
                min_value=30,
                max_value=220,
                step=1
            ),
            "Meal_Flag": st.column_config.SelectboxColumn(
                "Meal Flag",
                options=[0, 1],
                help="0 = No Meal, 1 = Meal"
            ),
        },
    )

    st.caption(
        "Enter the last 10 CGM readings. "
        "Meal_Flag: 0 = fasting/no meal, 1 = meal consumed."
    )

    col1, col2 = st.columns(2)
    with col1:
        carbs_limit = st.number_input(
            "Carbs Limit (g)",
            min_value=0,
            max_value=200,
            value=50
        )
    with col2:
        meal_type = st.selectbox(
            "Meal Type",
            ["breakfast", "lunch", "dinner"]
        )

    submitted = st.form_submit_button("Run CDSS Analysis", type="primary")

# Run Prediction
if submitted:
    st.session_state.cgm_data = entries.copy()
    
    if entries["Glucose"].isna().any():
        st.error("❌ Missing glucose values detected. Please fill all entries.")
        st.stop()
    
    if (entries["Glucose"] < 40).any() or (entries["Glucose"] > 400).any():
        st.error("❌ Glucose values outside clinical range (40-400 mg/dL).")
        st.stop()

    try:
        with st.spinner("🔄 Running CDSS Analysis with XGBoost..."):
            result = engine.run(
                entries=entries,
                carbs_limit=carbs_limit,
                meal_type=meal_type
            )

        # Current Status
        st.subheader("📊 Current Glucose Status")
        current_glucose = float(entries["Glucose"].iloc[-1])
        trend = entries["Glucose"].iloc[-1] - entries["Glucose"].iloc[-2] if len(entries) > 1 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status_color = "🟢" if 70 <= current_glucose <= 180 else "🔴" if current_glucose > 180 else "🟡"
            st.metric(
                "Current Glucose",
                f"{current_glucose:.0f} mg/dL",
                delta=f"{trend:+.1f} mg/dL",
                delta_color="inverse" if trend > 0 else "normal"
            )
        with col2:
            st.metric("Trend", result["trend"]["trend"])
        with col3:
            st.metric("Risk Level", result["trend"]["type"])

        # CGM Trend Plot
        st.subheader("📈 CGM Trend (Last 10 readings)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(len(entries)), entries["Glucose"], 'b-o', linewidth=2, label='Glucose')
        ax.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Hypo threshold (70)')
        ax.axhline(y=180, color='r', linestyle='--', alpha=0.5, label='Hyper threshold (180)')
        ax.fill_between(range(len(entries)), 70, 180, alpha=0.1, color='green')
        ax.set_xlabel('Time (5-min intervals)')
        ax.set_ylabel('Glucose (mg/dL)')
        ax.set_title('CGM Trend')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        # Glucose Forecast
        st.subheader("📈 Glucose Forecast (XGBoost Predictions)")
        preds = result["predictions"]

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "30 min",
                f"{preds['30min']['prediction']:.1f} mg/dL",
                delta=f"{preds['30min']['prediction'] - current_glucose:+.1f}",
                delta_color="inverse"
            )
            st.write(f"**Lower Bound:** {preds['30min']['lower']:.1f} mg/dL")
            st.write(f"**Upper Bound:** {preds['30min']['upper']:.1f} mg/dL")

        with c2:
            st.metric(
                "60 min",
                f"{preds['60min']['prediction']:.1f} mg/dL",
                delta=f"{preds['60min']['prediction'] - current_glucose:+.1f}",
                delta_color="inverse"
            )
            st.write(f"**Lower Bound:** {preds['60min']['lower']:.1f} mg/dL")
            st.write(f"**Upper Bound:** {preds['60min']['upper']:.1f} mg/dL")

        with c3:
            st.metric(
                "120 min",
                f"{preds['120min']['prediction']:.1f} mg/dL",
                delta=f"{preds['120min']['prediction'] - current_glucose:+.1f}",
                delta_color="inverse"
            )
            st.write(f"**Lower Bound:** {preds['120min']['lower']:.1f} mg/dL")
            st.write(f"**Upper Bound:** {preds['120min']['upper']:.1f} mg/dL")

        # Food Recommendations
        st.subheader("🍽️ Food Recommendations")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                pd.DataFrame(result["recommendation"]["foods"]),
                use_container_width=True,
                hide_index=True
            )
        with col2:
            st.info(f"**Strategy:** {result['recommendation']['strategy']}")
            st.info(f"**Foods Found:** {result['recommendation']['count']}")

        # Activity Recommendation
        st.subheader("🏃 Activity Recommendation")
        if current_glucose > 180:
            activity = "🚶 Light walking (10–15 min) recommended. Avoid intense exercise."
        elif current_glucose < 80:
            activity = "🍯 Consume quick glucose and rest. Monitor closely."
        elif 80 <= current_glucose <= 140:
            activity = "🏃 Moderate activity (20–30 min walk or cycling) recommended."
        else:
            activity = "🏃 Light to moderate activity recommended. Stay hydrated."
        st.success(activity)

        # Clinical Strategy
        st.subheader("🧠 Clinical Strategy")
        st.info(result["recommendation"]["strategy"])

        # Model Info
        with st.expander("ℹ️ Model Information"):
            model_info = engine.predictor.get_model_info()
            st.json(model_info)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e)

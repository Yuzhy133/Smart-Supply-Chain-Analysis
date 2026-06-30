import streamlit as st
import pandas as pd
import numpy as np

from data_processing import load_data, preprocess_data, get_basic_metrics, get_preprocessing_summary
from eda import render_eda_report

from modeling import (
    train_sales_models,
    model_performance_chart,
    prediction_scatter,
    feature_importance_table,
    feature_importance_chart,
    prophet_forecast,
    prophet_forecast_chart,
    analyze_prophet_components,
    prophet_component_charts,
    MODELING_OVERVIEW_ANALYSIS,
    MODEL_SELECTION_ANALYSIS,
    MODEL_TRAINING_EVALUATION_ANALYSIS,
    # MODEL_TUNING_ANALYSIS,
    # UNIQUE_FEATURE_SPLIT_ANALYSIS,
    # OVERFITTING_ANALYSIS,
    MODEL_PERFORMANCE_CONCLUSION,
    # FEATURE_IMPORTANCE_ANALYSIS,
    PROPHET_FORECAST_ANALYSIS,
    PROPHET_COMPONENT_ANALYSIS,
    # MODEL_LIMITATIONS_ANALYSIS
)


st.set_page_config(
    page_title="Supply Chain Analytics Dashboard",
    page_icon="📦",
    layout="wide"
)


def show_plot(fig, missing_message="Required columns are not available for this chart."):
    if fig is None:
        st.info(missing_message)
    elif hasattr(fig, "savefig"):
        st.pyplot(fig)
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_analysis(text, heading_level=4, drop_first_heading=False):
    lines = text.splitlines()
    if drop_first_heading:
        for index, line in enumerate(lines):
            if line.strip():
                if line.lstrip().startswith("#"):
                    lines.pop(index)
                break

    heading_levels = [
        len(line) - len(line.lstrip("#"))
        for line in lines
        if line.startswith("#")
    ]
    min_heading_level = min(heading_levels) if heading_levels else None
    rendered_lines = []

    for line in lines:
        if min_heading_level is not None and line.startswith("#"):
            current_level = len(line) - len(line.lstrip("#"))
            new_level = min(6, heading_level + current_level - min_heading_level)
            rendered_lines.append("#" * new_level + line[current_level:])
        else:
            rendered_lines.append(line)

    st.markdown("\n".join(rendered_lines).strip())


def format_money(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def format_number(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def main():
    st.title("Supply Chain Analysis and Optimization Dashboard")
    st.caption(
        "A modular Streamlit dashboard that reproduces EDA and modeling results from the dataset."
    )

    st.sidebar.header("Data Source")

    try:
        raw_data = load_data()
        data = preprocess_data(raw_data)
    except FileNotFoundError:
        st.error(
            "Dataset not found. Please place `DataCoSupplyChainDataset.csv` "
            "in the same folder as `run.py`."
        )
        st.stop()
    except Exception as error:
        st.error(f"Failed to load data: {error}")
        st.stop()

    st.sidebar.success(f"Loaded DataCoSupplyChain.csv: {len(data):,} rows and {data.shape[1]:,} columns")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        label="Navigation",
        options=[
            "Project Overview",
            "Exploratory Data Analysis",
            "Sales Prediction Modeling",
        ],
        label_visibility="collapsed"
    )

    if page == "Project Overview":
        show_overview(raw_data, data)

    elif page == "Exploratory Data Analysis":
        render_eda_report(data)

    elif page == "Sales Prediction Modeling":
        show_modeling(data)


def show_overview(raw_data, data):
    st.header("Project Overview")

    st.write(
        """
        This project analyzes supply chain performance from several perspectives:
        sales performance, product/category contribution, regional patterns,
        fraud risk, delivery delay risk, customer value, and sales prediction.
        """
    )

    metrics = get_basic_metrics(data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Records", f"{metrics['records']:,}")
    col2.metric("Columns", f"{metrics['columns']:,}")
    col3.metric("Total Sales", format_money(metrics["total_sales"]))
    col4.metric("Average Sales", format_money(metrics["average_sales"]))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Total Profit", format_money(metrics["total_profit"]))
    col6.metric(
        "Suspected Fraud Rate",
        "N/A" if pd.isna(metrics["fraud_rate"]) else f"{metrics['fraud_rate']:.2f}%"
    )
    col7.metric(
        "Late Delivery Rate",
        "N/A" if pd.isna(metrics["late_delivery_rate"]) else f"{metrics['late_delivery_rate']:.2f}%"
    )
    col8.empty()

    st.subheader("Preprocessing Summary")
    st.dataframe(get_preprocessing_summary(raw_data, data), use_container_width=True)

    st.subheader("Data Preview")
    st.dataframe(data.head(20), use_container_width=True)

    st.subheader("Column Summary")
    summary = pd.DataFrame({
        "column": data.columns,
        "dtype": data.dtypes.astype(str).values,
        "missing_values": data.isna().sum().values,
        "missing_rate": (data.isna().mean().values * 100).round(2)
    })
    st.dataframe(summary, use_container_width=True)


def show_eda(data):
    st.header("Exploratory Data Analysis")

    st.subheader("Sales Distribution")
    show_plot(sales_distribution(data))

    st.subheader("Monthly Sales Trend")
    show_plot(monthly_sales_trend(data))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top Products by Sales")
        show_plot(top_products_by_sales(data, n=10))
    with col2:
        st.subheader("Top Categories by Sales")
        show_plot(category_sales(data, n=10))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Sales by Market")
        show_plot(market_sales(data))
    with col4:
        st.subheader("Sales by Region")
        show_plot(region_sales(data, n=15))

    st.subheader("Profit Distribution")
    show_plot(profit_distribution(data))

    st.subheader("Loss-Making Categories or Products")
    show_plot(loss_categories(data, n=10))

    st.subheader("Correlation Heatmap")
    show_plot(correlation_heatmap(data))


def show_fraud_and_delivery_risk(data):
    st.header("Fraud and Delivery Risk Analysis")

    metrics = get_basic_metrics(data)

    col1, col2 = st.columns(2)
    col1.metric(
        "Suspected Fraud Rate",
        "N/A" if pd.isna(metrics["fraud_rate"]) else f"{metrics['fraud_rate']:.2f}%"
    )
    col2.metric(
        "Late Delivery Rate",
        "N/A" if pd.isna(metrics["late_delivery_rate"]) else f"{metrics['late_delivery_rate']:.2f}%"
    )

    st.subheader("Fraud Risk by Region")
    show_plot(fraud_by_column(data, "order_region"))

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Fraud Risk by Payment Type")
        show_plot(fraud_by_column(data, "type"))
    with col4:
        st.subheader("Fraud Risk by Category")
        show_plot(fraud_by_column(data, "category_name"))

    st.subheader("Late Delivery Risk by Region")
    show_plot(late_delivery_by_region(data))


def show_rfm(data):
    st.header("RFM Customer Segmentation")

    rfm = create_rfm_table(data)

    if rfm is None:
        st.info(
            "RFM analysis requires `customer_id`, `order_date_dateorders`, and `sales` columns."
        )
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Customers", f"{rfm['customer_id'].nunique():,}")
    col2.metric("Average Frequency", f"{rfm['frequency'].mean():.2f}")
    col3.metric("Average Monetary Value", format_money(rfm["monetary"].mean()))

    st.subheader("Customer Segment Distribution")
    show_plot(rfm_segment_chart(rfm))

    st.subheader("RFM Table")
    st.dataframe(rfm.head(50), use_container_width=True)


def show_modeling(data):
    st.header("Sales Prediction Modeling")
    show_analysis(MODELING_OVERVIEW_ANALYSIS, heading_level=3, drop_first_heading=True)
    show_analysis(MODEL_SELECTION_ANALYSIS, heading_level=3)

    st.write(
        """
        The models below are trained when this page runs. The performance table,
        best model, prediction plot, and feature importance are calculated from the current dataset.
        """
    )

    sample_size = st.slider(
        "Sample size for model training",
        min_value=5000,
        max_value=min(50000, len(data)),
        value=min(30000, len(data)),
        step=5000
    )

    with st.spinner("Training models from the current dataset..."):
        try:
            model_result = train_sales_models(data, sample_size=sample_size)
        except Exception as error:
            st.error(f"Modeling failed: {error}")
            return

    results = model_result["results"]

    st.subheader("Model Features and Target")
    st.write("Target variable:", model_result["target"])
    st.write("Feature variables:", ", ".join(model_result["features"]))
    # show_analysis(MODEL_TUNING_ANALYSIS, heading_level=4)
    # show_analysis(UNIQUE_FEATURE_SPLIT_ANALYSIS, heading_level=4)

    st.subheader("Model Performance")
    show_analysis(MODEL_TRAINING_EVALUATION_ANALYSIS, heading_level=4)
    st.dataframe(
        results.style.format({
            "R2": "{:.4f}",
            "MAE": "{:,.2f}",
            "MSE": "{:,.2f}",
            "RMSE": "{:,.2f}"
        }),
        use_container_width=True
    )

    show_plot(model_performance_chart(results))
    show_analysis(MODEL_PERFORMANCE_CONCLUSION, heading_level=4)

    best_model = results.iloc[0]["Model"]
    best_r2 = results.iloc[0]["R2"]
    best_mae = results.iloc[0]["MAE"]

    st.success(
        f"Best model based on R²: {best_model}. "
        f"R² = {best_r2:.4f}, MAE = {best_mae:,.2f}."
    )

    st.subheader("Actual vs Predicted")
    selected_model = st.selectbox(
        "Select model for prediction scatter plot",
        results["Model"].tolist()
    )
    show_plot(
        prediction_scatter(model_result["predictions"][selected_model], selected_model)
    )
    # show_analysis(OVERFITTING_ANALYSIS, heading_level=4)

    st.subheader("Feature Importance")
    importance = feature_importance_table(model_result, selected_model)

    if importance is None:
        st.info("The selected model does not provide feature_importances_. Try Random Forest or Gradient Boosting.")
    else:
        show_plot(feature_importance_chart(importance, n=15))
        st.dataframe(importance.head(30), use_container_width=True)
        # show_analysis(FEATURE_IMPORTANCE_ANALYSIS, heading_level=4)

    st.divider()
    st.subheader("Prophet Time Series Forecasting")

    st.write(
        """
        This part follows the original notebook logic. Sales are aggregated by order date,
        then Prophet is used to forecast the next 30 days.
        """
    )

    try:
        with st.spinner("Training Prophet model..."):
            prophet_model, forecast, daily_sales, prophet_results = prophet_forecast(data)

        col1, col2 = st.columns(2)
        col1.metric("Prophet MAE", f"{prophet_results['MAE']:,.2f}")
        col2.metric("Prophet MAPE", f"{prophet_results['MAPE']:.2f}%")

        st.subheader("Sales Forecast with Prophet")
        show_plot(prophet_forecast_chart(forecast, daily_sales))
        show_analysis(PROPHET_FORECAST_ANALYSIS, heading_level=4)

        st.subheader("Prophet Component Analysis")

        component_df, config_df = analyze_prophet_components(
            prophet_model,
            forecast
        )

        if config_df is not None and not config_df.empty:
            st.write("Model configuration information")
            st.dataframe(config_df, use_container_width=True)

        if component_df is not None and not component_df.empty:
            st.write("Analysis of the impact of each component")
            st.dataframe(component_df, use_container_width=True)

        component_fig = prophet_component_charts(prophet_model, forecast)

        if component_fig is not None:
            st.pyplot(component_fig)
            show_analysis(PROPHET_COMPONENT_ANALYSIS, heading_level=4)

        # show_analysis(MODEL_LIMITATIONS_ANALYSIS, heading_level=4)

    except ImportError:
        st.warning(
            "Prophet is not installed. Please run `pip install prophet` first."
        )

    except Exception as error:
        st.error(f"Prophet forecasting failed: {error}")



if __name__ == "__main__":
    main()

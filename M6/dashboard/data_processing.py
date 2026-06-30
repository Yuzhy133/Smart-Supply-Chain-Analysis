import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data(show_spinner=False)
def load_data(uploaded_file=None, local_path="DataCoSupplyChainDataset.csv"):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file, encoding="latin1")

    return pd.read_csv(local_path, encoding="latin1")


def standardize_column_names(data):
    data = data.copy()
    data.columns = (
        data.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[()]", "", regex=True)
    )
    return data


def convert_date_columns(data):
    data = data.copy()

    if "order_date_dateorders" in data.columns:
        data["order_date_dateorders"] = pd.to_datetime(
            data["order_date_dateorders"],
            errors="coerce"
        )

    if "shipping_date_dateorders" in data.columns:
        data["shipping_date_dateorders"] = pd.to_datetime(
            data["shipping_date_dateorders"],
            errors="coerce"
        )

    return data


def filter_by_order_date(data, cutoff_date="2017-10-01"):
    data = data.copy()

    if "order_date_dateorders" in data.columns:
        cutoff_date = pd.Timestamp(cutoff_date)
        data = data[data["order_date_dateorders"] < cutoff_date].copy()

    return data


def drop_unnecessary_columns(data):
    data = data.copy()

    columns_to_drop = [
        "product_description",
        "order_zipcode",
        "customer_zipcode",
        "customer_email",
        "customer_password"
    ]

    data = data.drop(
        columns=[col for col in columns_to_drop if col in data.columns],
        errors="ignore"
    )

    return data


def engineer_time_features(data):
    data = data.copy()

    if "order_date_dateorders" in data.columns:
        data["ORDER_YEAR"] = data["order_date_dateorders"].dt.year
        data["ORDER_MONTH"] = data["order_date_dateorders"].dt.month
        data["ORDER_DAY"] = data["order_date_dateorders"].dt.day
        data["ORDER_WEEKDAY"] = data["order_date_dateorders"].dt.weekday
        data["ORDER_DATE"] = data["order_date_dateorders"].dt.date
        data["ORDER_YEAR_MONTH"] = data["order_date_dateorders"].dt.to_period("M").astype(str)

        # Lowercase aliases are kept only for compatibility with app modules.
        # The main preprocessing output still follows the uploaded notebook.
        data["order_year"] = data["ORDER_YEAR"]
        data["order_month"] = data["ORDER_MONTH"]
        data["order_day"] = data["ORDER_DAY"]
        data["order_weekday"] = data["ORDER_WEEKDAY"]
        data["order_date"] = data["ORDER_DATE"]
        data["order_year_month"] = data["ORDER_YEAR_MONTH"]

    return data


def engineer_risk_features(data):
    data = data.copy()

    if {"days_for_shipping_real", "days_for_shipment_scheduled"}.issubset(data.columns):
        data["calculated_shipping_delay"] = (
            data["days_for_shipping_real"] - data["days_for_shipment_scheduled"]
        )
        data["is_late_delivery"] = (
            data["days_for_shipping_real"] > data["days_for_shipment_scheduled"]
        ).astype(int)

    if {"shipping_date_dateorders", "order_date_dateorders"}.issubset(data.columns):
        data["calculated_shipping_days"] = (
            data["shipping_date_dateorders"] - data["order_date_dateorders"]
        ).dt.days

    if "order_status" in data.columns:
        data["is_suspected_fraud"] = (
            data["order_status"].astype(str).str.upper() == "SUSPECTED_FRAUD"
        ).astype(int)

    return data


def preprocess_data(data):
    data = standardize_column_names(data)
    data = convert_date_columns(data)
    data = filter_by_order_date(data, cutoff_date="2017-10-01")
    data = drop_unnecessary_columns(data)
    data = engineer_time_features(data)
    data = engineer_risk_features(data)

    return data


def get_preprocessing_summary(raw_data, processed_data):
    removed_rows = len(raw_data) - len(processed_data)
    removed_columns = set(standardize_column_names(raw_data).columns) - set(processed_data.columns)

    summary = pd.DataFrame({
        "Step": [
            "Original records",
            "Processed records",
            "Removed records",
            "Original columns",
            "Processed columns",
            "Dropped columns"
        ],
        "Result": [
            f"{len(raw_data):,}",
            f"{len(processed_data):,}",
            f"{removed_rows:,}",
            f"{raw_data.shape[1]:,}",
            f"{processed_data.shape[1]:,}",
            ", ".join(sorted(removed_columns)) if removed_columns else "None"
        ]
    })

    return summary


def get_basic_metrics(data):
    metrics = {
        "records": len(data),
        "columns": data.shape[1],
        "total_sales": np.nan,
        "average_sales": np.nan,
        "total_profit": np.nan,
        "fraud_rate": np.nan,
        "late_delivery_rate": np.nan,
    }

    if "sales" in data.columns:
        metrics["total_sales"] = data["sales"].sum()
        metrics["average_sales"] = data["sales"].mean()

    if "benefit_per_order" in data.columns:
        metrics["total_profit"] = data["benefit_per_order"].sum()
    elif "order_profit_per_order" in data.columns:
        metrics["total_profit"] = data["order_profit_per_order"].sum()

    if "is_suspected_fraud" in data.columns:
        metrics["fraud_rate"] = data["is_suspected_fraud"].mean() * 100

    if "is_late_delivery" in data.columns:
        metrics["late_delivery_rate"] = data["is_late_delivery"].mean() * 100
    elif "late_delivery_risk" in data.columns:
        metrics["late_delivery_rate"] = data["late_delivery_risk"].mean() * 100

    return metrics


def available_columns(data, columns):
    return [col for col in columns if col in data.columns]

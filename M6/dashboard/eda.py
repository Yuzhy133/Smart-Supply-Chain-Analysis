import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# Helper functions
def _exists(data, cols):
    return all(col in data.columns for col in cols)


def _section(title, description=None):
    st.markdown(f"### {title}")
    if description:
        st.caption(description)


def _subsection(title, description=None):
    st.markdown(f"#### {title}")
    if description:
        st.caption(description)


def _show_fig(fig, height=None):
    if fig is None:
        st.info("Required columns are not available for this visualization.")
    else:
        if height is not None:
            fig.update_layout(height=height)
        st.plotly_chart(fig, use_container_width=True)


def _show_table(df, height=None):
    if df is None or len(df) == 0:
        st.info("No table can be generated from the current data.")
    else:
        st.dataframe(df, use_container_width=True, height=height)


def _format_money(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def _format_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2f}%"


def _clean_for_plot(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


# Hypotheses and reusable summary helpers
def get_eda_summary(data):
    summary = {
        "Rows after preprocessing": f"{data.shape[0]:,}",
        "Columns after preprocessing": f"{data.shape[1]:,}",
    }

    if "order_date_dateorders" in data.columns:
        order_dates = pd.to_datetime(data["order_date_dateorders"], errors="coerce")
        if order_dates.notna().any():
            summary["Date range"] = f"{order_dates.min().date()} to {order_dates.max().date()}"

    if "customer_id" in data.columns:
        summary["Unique customers"] = f"{data['customer_id'].nunique():,}"

    if "product_name" in data.columns:
        summary["Unique products"] = f"{data['product_name'].nunique():,}"

    if "market" in data.columns:
        summary["Markets"] = f"{data['market'].nunique():,}"

    describe_cols = [
        "sales",
        "sales_per_customer",
        "benefit_per_order",
        "order_item_quantity",
        "days_for_shipping_real",
        "days_for_shipment_scheduled",
        "late_delivery_risk",
    ]

    available_cols = [col for col in describe_cols if col in data.columns]
    describe_table = data[available_cols].describe().T if available_cols else pd.DataFrame()

    return summary, describe_table


def original_hypotheses_table():
    return pd.DataFrame(
        {
            "No.": [1, 2, 3],
            "Original hypothesis": [
                "Products with higher sales volume may contribute more to overall sales and business performance.",
                "Some regions, markets, or shipping modes may have higher late-delivery risk.",
                "Fraud risk and loss may be concentrated in specific regions, payment methods, product categories, or customers.",
            ],
            "EDA purpose": [
                "Understand product contribution and sales distribution.",
                "Identify logistics delay patterns.",
                "Identify operational risk patterns before modeling.",
            ],
        }
    )


def revised_hypotheses_table():
    return pd.DataFrame(
        {
            "No.": [1, 2, 3],
            "Revised hypothesis": [
                "Sales performance is unevenly distributed across products, markets, and regions.",
                "Late delivery risk differs across region, market, and shipping mode combinations.",
                "Fraud and loss patterns are not random; they are concentrated in certain operational groups.",
            ],
            "How EDA supports it": [
                "Use product quantity, market sales, regional sales, and monthly trend analysis.",
                "Use late-delivery rate by region, shipping mode, market, and their combinations.",
                "Use suspected-fraud charts, loss-making categories, and customer-level fraud concentration.",
            ],
        }
    )


# Univariate analysis
def sales_distribution(data):
    if "sales" not in data.columns:
        return None

    plot_sales = data[["sales"]].copy()
    plot_sales = plot_sales[plot_sales["sales"] > 0].copy()
    if plot_sales.empty:
        return None

    plot_sales["log_sales"] = np.log10(plot_sales["sales"])

    mean_sales = data["sales"].mean()
    median_sales = data["sales"].median()

    fig = px.histogram(
        plot_sales,
        x="log_sales",
        nbins=60,
        marginal="box",
        title="Distribution of Order Sales with Outliers Retained",
        labels={"log_sales": "Sales per order item, log10 scale"},
        opacity=0.78,
    )

    fig.add_vline(
        x=np.log10(median_sales),
        line_dash="dash",
        annotation_text=f"Median: ${median_sales:,.0f}",
        annotation_position="top left",
    )
    fig.add_vline(
        x=np.log10(mean_sales),
        line_dash="dot",
        annotation_text=f"Mean: ${mean_sales:,.0f}",
        annotation_position="top right",
    )

    sales_ticks = np.array([10, 25, 50, 100, 200, 500, 1000, 2000])
    fig.update_xaxes(
        tickvals=np.log10(sales_ticks),
        ticktext=[f"${x:,.0f}" for x in sales_ticks],
    )
    fig.update_layout(
        yaxis_title="Number of order items",
        bargap=0.03,
        height=520,
    )

    return fig


def sales_distribution_stats(data):
    if "sales" not in data.columns:
        return pd.DataFrame()

    q1 = data["sales"].quantile(0.25)
    q3 = data["sales"].quantile(0.75)
    iqr = q3 - q1
    outlier_threshold = q3 + 1.5 * iqr
    outlier_share = (data["sales"] > outlier_threshold).mean() * 100

    return pd.DataFrame(
        {
            "Metric": [
                "Mean sales",
                "Median sales",
                "Upper outlier threshold using 1.5*IQR rule",
                "Share above this threshold",
            ],
            "Value": [
                data["sales"].mean(),
                data["sales"].median(),
                outlier_threshold,
                outlier_share,
            ],
        }
    )


def profit_distribution(data):
    if "benefit_per_order" not in data.columns:
        return None

    benefit_lower = data["benefit_per_order"].quantile(0.01)
    benefit_upper = data["benefit_per_order"].quantile(0.99)
    plot_benefit = data[data["benefit_per_order"].between(benefit_lower, benefit_upper)].copy()

    if plot_benefit.empty:
        return None

    fig = px.histogram(
        plot_benefit,
        x="benefit_per_order",
        nbins=60,
        title="Distribution of Benefit Per Order, Trimmed at the 1st and 99th Percentiles",
        labels={"benefit_per_order": "Benefit per order"},
        opacity=0.78,
    )

    fig.add_vline(
        x=0,
        line_dash="dash",
        annotation_text="Break-even",
        annotation_position="top left",
    )
    fig.add_vline(
        x=data["benefit_per_order"].median(),
        line_dash="dot",
        annotation_text=f"Median: ${data['benefit_per_order'].median():,.0f}",
        annotation_position="top right",
    )

    fig.update_layout(
        yaxis_title="Number of order items",
        height=520,
    )

    return fig


def profit_distribution_stats(data):
    if "benefit_per_order" not in data.columns:
        return pd.DataFrame()

    loss_rate = (data["benefit_per_order"] < 0).mean() * 100
    return pd.DataFrame(
        {
            "Metric": ["Share of order items with negative benefit"],
            "Value": [loss_rate],
        }
    )


def top_products_by_sales(data, n=10):
    if not _exists(data, ["product_name", "order_item_quantity"]):
        return None

    top_products = (
        data.groupby("product_name", as_index=False)["order_item_quantity"]
        .sum()
        .sort_values("order_item_quantity", ascending=False)
        .head(n)
        .sort_values("order_item_quantity", ascending=True)
    )

    fig = px.bar(
        top_products,
        x="order_item_quantity",
        y="product_name",
        orientation="h",
        title=f"Top {n} Best-Selling Products by Quantity",
        labels={
            "order_item_quantity": "Total quantity sold",
            "product_name": "Product",
        },
        text="order_item_quantity",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(height=520)

    return fig


def market_sales(data):
    if not _exists(data, ["market", "sales_per_customer"]):
        return None

    market_sum = (
        data.groupby("market", as_index=False)["sales_per_customer"]
        .sum()
        .sort_values("sales_per_customer", ascending=False)
    )
    market_sum["sales_million"] = market_sum["sales_per_customer"] / 1e6

    fig = px.bar(
        market_sum,
        x="market",
        y="sales_million",
        title="Total Sales of All Markets",
        labels={
            "market": "Market",
            "sales_million": "Sales, millions",
        },
        text="sales_million",
    )
    fig.update_traces(texttemplate="%{text:.2f}M", textposition="outside")
    fig.update_layout(height=460)

    return fig


def region_sales(data, n=None):
    if not _exists(data, ["order_region", "sales_per_customer"]):
        return None

    region_sum = (
        data.groupby("order_region", as_index=False)["sales_per_customer"]
        .sum()
        .sort_values("sales_per_customer", ascending=True)
    )

    if n is not None:
        region_sum = region_sum.tail(n)

    region_sum["sales_million"] = region_sum["sales_per_customer"] / 1e6

    fig = px.bar(
        region_sum,
        x="sales_million",
        y="order_region",
        orientation="h",
        title="Total Sales in Each Region",
        labels={
            "sales_million": "Sales, millions",
            "order_region": "Region",
        },
        text="sales_million",
    )
    fig.update_traces(texttemplate="%{text:.2f}M", textposition="outside")
    fig.update_layout(height=max(520, 25 * len(region_sum)))

    return fig


def loss_categories(data, n=10):
    if not _exists(data, ["benefit_per_order", "category_name"]):
        return None

    loss = data[data["benefit_per_order"] < 0]
    if loss.empty:
        return None

    loss_products = (
        loss["category_name"]
        .value_counts()
        .head(n)
        .reset_index()
    )
    loss_products.columns = ["category_name", "lost_order_count"]
    loss_products = loss_products.sort_values("lost_order_count", ascending=True)

    fig = px.bar(
        loss_products,
        x="lost_order_count",
        y="category_name",
        orientation="h",
        title=f"Top {n} Product Categories with the Most Loss-Making Orders",
        labels={
            "lost_order_count": "Number of loss-making orders",
            "category_name": "Product category",
        },
        text="lost_order_count",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(height=520)

    return fig


# Fraud analysis
def _fraud_orders(data):
    if "order_status" not in data.columns:
        return pd.DataFrame()
    return data[data["order_status"] == "SUSPECTED_FRAUD"].copy()


def fraud_by_region(data):
    if not _exists(data, ["order_status", "order_region"]):
        return None

    fraud_orders = _fraud_orders(data)
    if fraud_orders.empty:
        return None

    fraud_region_count = fraud_orders["order_region"].value_counts()
    fraud_region_pct = fraud_region_count / fraud_region_count.sum() * 100

    fraud_region_df = fraud_region_pct.reset_index()
    fraud_region_df.columns = ["order_region", "fraud_pct"]
    fraud_region_df = fraud_region_df.sort_values("fraud_pct", ascending=True)

    fig = px.bar(
        fraud_region_df,
        x="fraud_pct",
        y="order_region",
        orientation="h",
        title="Proportion of Suspected Fraud Orders by Region",
        labels={
            "fraud_pct": "Proportion of suspected fraud orders (%)",
            "order_region": "Region",
        },
        text="fraud_pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=max(520, 25 * len(fraud_region_df)))
    fig.update_xaxes(range=[0, 100])

    return fig


def fraud_payment_distribution(data):
    if not _exists(data, ["order_status", "type"]):
        return None

    fraud_orders = _fraud_orders(data)
    if fraud_orders.empty:
        return None

    fraud_payment = fraud_orders["type"].value_counts().reset_index()
    fraud_payment.columns = ["type", "fraud_order_count"]
    fraud_payment = fraud_payment.sort_values("fraud_order_count", ascending=False)

    fig = px.bar(
        fraud_payment,
        x="type",
        y="fraud_order_count",
        title="Distribution of Payment Methods Suspected of Fraud",
        labels={
            "type": "Payment method",
            "fraud_order_count": "Number of suspected fraud orders",
        },
        text="fraud_order_count",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(height=440)

    return fig


def fraud_category_distribution(data, n=10):
    if not _exists(data, ["order_status", "category_name"]):
        return None

    fraud_orders = _fraud_orders(data)
    if fraud_orders.empty:
        return None

    fraud_cat = fraud_orders["category_name"].value_counts().head(n).reset_index()
    fraud_cat.columns = ["category_name", "fraud_order_count"]
    fraud_cat = fraud_cat.sort_values("fraud_order_count", ascending=True)

    fig = px.bar(
        fraud_cat,
        x="fraud_order_count",
        y="category_name",
        orientation="h",
        title=f"Top {n} Product Categories Most Frequently Involved in Fraud",
        labels={
            "category_name": "Product category",
            "fraud_order_count": "Number of suspected fraud orders",
        },
        text="fraud_order_count",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(height=520)

    return fig


def fraud_customer_distribution(data, n=10):
    required = ["order_status", "customer_fname", "customer_lname"]
    if not _exists(data, required):
        return None

    temp = data.copy()
    temp["customer_full_name"] = (
        temp["customer_fname"].astype(str) + " " + temp["customer_lname"].astype(str)
    )

    fraud_orders = temp[temp["order_status"] == "SUSPECTED_FRAUD"]
    if fraud_orders.empty:
        return None

    fraud_customers = (
        fraud_orders["customer_full_name"]
        .value_counts()
        .head(n)
        .reset_index()
    )
    fraud_customers.columns = ["customer_full_name", "fraud_order_count"]
    fraud_customers = fraud_customers.sort_values("fraud_order_count", ascending=True)

    fig = px.bar(
        fraud_customers,
        x="fraud_order_count",
        y="customer_full_name",
        orientation="h",
        title=f"Top {n} Customers with the Most Suspected Fraud Cases",
        labels={
            "customer_full_name": "Customer",
            "fraud_order_count": "Number of suspected fraud orders",
        },
        text="fraud_order_count",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(height=520)

    return fig


def fraud_by_column(data, column):
    if column == "order_region":
        return fraud_by_region(data)
    if column == "type":
        return fraud_payment_distribution(data)
    if column == "category_name":
        return fraud_category_distribution(data)

    if not _exists(data, ["order_status", column]):
        return None

    fraud_orders = _fraud_orders(data)
    if fraud_orders.empty:
        return None

    fraud_count = fraud_orders[column].value_counts().head(10).reset_index()
    fraud_count.columns = [column, "fraud_order_count"]
    fraud_count = fraud_count.sort_values("fraud_order_count", ascending=True)

    fig = px.bar(
        fraud_count,
        x="fraud_order_count",
        y=column,
        orientation="h",
        title=f"Suspected Fraud Distribution by {column}",
        text="fraud_order_count",
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")

    return fig


# RFM analysis
def create_rfm_table(data):
    required = ["customer_id", "order_date_dateorders", "sales"]
    if not _exists(data, required):
        return None

    temp = data.copy()
    temp["order_date_dateorders"] = pd.to_datetime(temp["order_date_dateorders"], errors="coerce")
    temp = temp.dropna(subset=["customer_id", "order_date_dateorders", "sales"])

    if temp.empty:
        return None

    current_date = temp["order_date_dateorders"].max()

    rfm_data = temp[["customer_id", "order_date_dateorders", "sales"]].copy()
    rfm_data.columns = ["customer_id", "order_date", "sales"]

    recency_df = rfm_data.groupby("customer_id").order_date.max().reset_index()
    recency_df["recency"] = (current_date - recency_df["order_date"]).dt.days

    frequency_df = rfm_data.groupby("customer_id").order_date.count().reset_index()
    frequency_df.columns = ["customer_id", "frequency"]

    monetary_df = rfm_data.groupby("customer_id").sales.sum().reset_index()
    monetary_df.columns = ["customer_id", "monetary"]

    rfm_df = (
        recency_df
        .merge(frequency_df, on="customer_id")
        .merge(monetary_df, on="customer_id")
    )

    rfm_df["R_Score"] = pd.qcut(
        rfm_df["recency"],
        q=2,
        labels=[2, 1],
        duplicates="drop",
    )
    rfm_df["F_Score"] = pd.qcut(
        rfm_df["frequency"],
        q=2,
        labels=[1, 2],
        duplicates="drop",
    )
    rfm_df["M_Score"] = pd.qcut(
        rfm_df["monetary"],
        q=2,
        labels=[1, 2],
        duplicates="drop",
    )

    rfm_df["RFM_Score"] = (
        rfm_df["R_Score"].astype(str)
        + rfm_df["F_Score"].astype(str)
        + rfm_df["M_Score"].astype(str)
    )

    customer_segments = {
        "222": "Important Value Customer",
        "122": "Important to retain customers",
        "212": "Important development client",
        "112": "Important to retain customers",
        "221": "Regular Value Customer",
        "121": "Regularly maintain customers",
        "211": "General customer development",
        "111": "Ordinary customer retention",
    }

    rfm_df["Customer_Segment"] = rfm_df["RFM_Score"].map(customer_segments)
    rfm_df["customer_segment"] = rfm_df["Customer_Segment"]
    rfm_df["rfm_score"] = rfm_df["RFM_Score"]

    return rfm_df


def rfm_value_pyramid(rfm_df):
    if rfm_df is None or rfm_df.empty or "Customer_Segment" not in rfm_df.columns:
        return None

    value_order = [
        "Important Value Customer",
        "Important to retain customers",
        "Important development client",
        "Regular Value Customer",
        "Regularly maintain customers",
        "General customer development",
        "Ordinary customer retention",
    ]

    segment_counts = rfm_df["Customer_Segment"].value_counts()
    pyramid = pd.DataFrame(
        {
            "Customer_Segment": value_order,
            "customer_count": [segment_counts.get(seg, 0) for seg in value_order],
        }
    )
    pyramid["percentage"] = pyramid["customer_count"] / pyramid["customer_count"].sum() * 100
    pyramid = pyramid.sort_values("percentage", ascending=True)

    fig = px.bar(
        pyramid,
        x="percentage",
        y="Customer_Segment",
        orientation="h",
        title="Customer Value Pyramid",
        labels={
            "percentage": "Percentage of customers (%)",
            "Customer_Segment": "Customer segment",
        },
        text="customer_count",
        hover_data=["customer_count", "percentage"],
    )
    fig.update_traces(texttemplate="%{x:.1f}% (%{text})", textposition="outside")
    fig.update_layout(height=520)
    fig.update_xaxes(range=[0, 100])

    return fig


def rfm_segment_chart(rfm_df):
    if rfm_df is None or rfm_df.empty or "Customer_Segment" not in rfm_df.columns:
        return None

    segment_counts = rfm_df["Customer_Segment"].value_counts().reset_index()
    segment_counts.columns = ["Customer_Segment", "count"]

    fig = px.pie(
        segment_counts,
        names="Customer_Segment",
        values="count",
        title="Customer Segmentation Distribution Chart",
        hole=0.35,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=560)

    return fig


# Bivariate analysis
def monthly_sales_trend(data):
    if not _exists(data, ["order_date_dateorders", "sales"]):
        return None

    temp = data.copy()
    temp["order_date_dateorders"] = pd.to_datetime(temp["order_date_dateorders"], errors="coerce")
    temp = temp.dropna(subset=["order_date_dateorders"])

    daily_orders = temp.groupby(["order_date_dateorders"], as_index=False)["sales"].sum()
    daily_orders["year"] = daily_orders["order_date_dateorders"].dt.year
    daily_orders["month"] = daily_orders["order_date_dateorders"].dt.month

    monthly_orders_by_year = (
        daily_orders.groupby(["year", "month"], as_index=False)["sales"].sum()
    )
    monthly_orders_by_year["month_label"] = monthly_orders_by_year["month"].map(
        {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
    )

    fig = px.line(
        monthly_orders_by_year,
        x="month",
        y="sales",
        color="year",
        markers=True,
        title="Monthly Sales by Year",
        labels={
            "month": "Month",
            "sales": "Total sales",
            "year": "Year",
        },
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(1, 13)),
        ticktext=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    )
    fig.update_layout(height=520)

    return fig


def hourly_order_trend(data):
    if "order_date_dateorders" not in data.columns:
        return None

    temp = data.copy()
    temp["order_date_dateorders"] = pd.to_datetime(temp["order_date_dateorders"], errors="coerce")
    temp = temp.dropna(subset=["order_date_dateorders"])
    temp["hour"] = temp["order_date_dateorders"].dt.hour

    hourly_orders = temp.groupby("hour").size().reset_index(name="order_count")

    fig = px.line(
        hourly_orders,
        x="hour",
        y="order_count",
        markers=True,
        title="Variation in Number of Orders over 24 Hours",
        labels={
            "hour": "Hour",
            "order_count": "Number of orders",
        },
    )
    fig.update_xaxes(tickmode="linear", tick0=0, dtick=1)
    fig.update_layout(height=480)
    fig.update_yaxes(range=[6000, hourly_orders["order_count"].max() * 1.1])

    return fig


def late_delivery_by_region(data):
    if not _exists(data, ["order_region", "delivery_status"]):
        return None

    total_by_region = data["order_region"].value_counts()
    late_by_region = data[data["delivery_status"] == "Late delivery"]["order_region"].value_counts()

    late_rate = (late_by_region / total_by_region).fillna(0).sort_values(ascending=True)
    late_df = late_rate.reset_index()
    late_df.columns = ["order_region", "late_rate"]
    late_df["late_rate_pct"] = late_df["late_rate"] * 100

    fig = px.bar(
        late_df,
        x="late_rate_pct",
        y="order_region",
        orientation="h",
        title="Delay Rate of Deliveries by Region",
        labels={
            "late_rate_pct": "Delay rate (%)",
            "order_region": "Region",
        },
        text="late_rate_pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=max(520, 25 * len(late_df)))
    fig.update_xaxes(range=[0, 100])

    return fig


def late_delivery_risk_by_mode_and_market(data):
    required = ["shipping_mode", "market", "late_delivery_risk"]
    if not _exists(data, required):
        return None

    mode_count = (
        data.groupby(["shipping_mode", "late_delivery_risk"])
        .size()
        .reset_index(name="count")
    )
    mode_total = (
        data.groupby("shipping_mode")
        .size()
        .reset_index(name="total")
    )
    mode_late = mode_count.merge(mode_total, on="shipping_mode", how="left")
    mode_late["share"] = mode_late["count"] / mode_late["total"]
    mode_late["risk_label"] = mode_late["late_delivery_risk"].map(
        {0: "No late risk", 1: "Late risk"}
    )

    market_count = (
        data.groupby(["market", "late_delivery_risk"])
        .size()
        .reset_index(name="count")
    )
    market_total = (
        data.groupby("market")
        .size()
        .reset_index(name="total")
    )
    market_late = market_count.merge(market_total, on="market", how="left")
    market_late["share"] = market_late["count"] / market_late["total"]
    market_late["risk_label"] = market_late["late_delivery_risk"].map(
        {0: "No late risk", 1: "Late risk"}
    )

    fig = go.Figure()

    for risk_label in ["No late risk", "Late risk"]:
        tmp = mode_late[mode_late["risk_label"] == risk_label]
        fig.add_trace(
            go.Bar(
                x=tmp["shipping_mode"],
                y=tmp["share"] * 100,
                name=risk_label,
                visible=True,
                hovertemplate="Shipping mode=%{x}<br>Share=%{y:.1f}%<extra></extra>",
            )
        )

    for risk_label in ["No late risk", "Late risk"]:
        tmp = market_late[market_late["risk_label"] == risk_label]
        fig.add_trace(
            go.Bar(
                x=tmp["market"],
                y=tmp["share"] * 100,
                name=risk_label,
                visible=False,
                hovertemplate="Market=%{x}<br>Share=%{y:.1f}%<extra></extra>",
            )
        )

    fig.update_layout(
        title="Late Delivery Risk by Shipping Mode",
        xaxis_title="Shipping mode",
        yaxis_title="Share of order items (%)",
        barmode="stack",
        height=580,
        legend_title="Late delivery risk",
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Shipping mode",
                        "method": "update",
                        "args": [
                            {"visible": [True, True, False, False]},
                            {
                                "xaxis": {"title": "Shipping mode"},
                                "title": "Late Delivery Risk by Shipping Mode",
                            },
                        ],
                    },
                    {
                        "label": "Market",
                        "method": "update",
                        "args": [
                            {"visible": [False, False, True, True]},
                            {
                                "xaxis": {"title": "Market"},
                                "title": "Late Delivery Risk by Market",
                            },
                        ],
                    },
                ],
                "direction": "down",
                "showactive": True,
                "x": -0.15,
                "xanchor": "left",
                "y": 1.10,
                "yanchor": "top",
            }
        ],
    )
    fig.update_yaxes(range=[0, 100])

    return fig


def sales_benefit_scatter(data):
    required = ["sales", "benefit_per_order"]
    if not _exists(data, required):
        return None

    plot_data = data.copy()

    if len(plot_data) > 25000:
        plot_data = plot_data.sample(n=25000, random_state=42)

    sales_limit = data["sales"].quantile(0.99)
    benefit_low = data["benefit_per_order"].quantile(0.01)
    benefit_high = data["benefit_per_order"].quantile(0.99)

    plot_data = plot_data[
        (plot_data["sales"] <= sales_limit)
        & (plot_data["benefit_per_order"].between(benefit_low, benefit_high))
    ].copy()

    color_col = "customer_segment" if "customer_segment" in plot_data.columns else None

    fig = px.scatter(
        plot_data,
        x="sales",
        y="benefit_per_order",
        color=color_col,
        title="Sales vs Benefit Per Order",
        labels={
            "sales": "Sales per order item",
            "benefit_per_order": "Benefit per order",
            "customer_segment": "Customer segment",
        },
        opacity=0.45,
        hover_data=[col for col in ["category_name", "market", "order_region"] if col in plot_data.columns],
    )
    fig.add_hline(y=0, line_dash="dash", annotation_text="Break-even")
    fig.update_layout(height=560)

    return fig


def sales_benefit_correlation(data):
    if not _exists(data, ["sales", "benefit_per_order"]):
        return None

    return data[["sales", "benefit_per_order"]].corr().iloc[0, 1]


def loss_rate_by_region(data):
    if not _exists(data, ["benefit_per_order", "order_region"]):
        return None

    loss = data[data["benefit_per_order"] < 0]
    if loss.empty:
        return None

    loss_counts = loss["order_region"].value_counts()
    total_counts = data["order_region"].value_counts()

    loss_rate = (loss_counts / total_counts).fillna(0).sort_values(ascending=True)
    loss_rate_df = loss_rate.reset_index()
    loss_rate_df.columns = ["order_region", "loss_rate"]
    loss_rate_df["loss_rate_pct"] = loss_rate_df["loss_rate"] * 100

    fig = px.bar(
        loss_rate_df,
        x="loss_rate_pct",
        y="order_region",
        orientation="h",
        title="Loss Rates by Region",
        labels={
            "loss_rate_pct": "Loss rate (%)",
            "order_region": "Region",
        },
        text="loss_rate_pct",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=max(520, 25 * len(loss_rate_df)))
    fig.update_xaxes(range=[0, 100])

    return fig


def loss_rate_by_region_stats(data):
    if not _exists(data, ["benefit_per_order", "order_region"]):
        return pd.DataFrame()

    loss = data[data["benefit_per_order"] < 0]
    if loss.empty:
        return pd.DataFrame()

    loss_counts = loss["order_region"].value_counts()
    total_counts = data["order_region"].value_counts()
    loss_rate = (loss_counts / total_counts).fillna(0).sort_values(ascending=False)

    return pd.DataFrame(
        {
            "Metric": [
                "Total number of loss-making orders",
                "Total order loss amount",
                "Highest regional loss rate",
            ],
            "Value": [
                len(loss),
                loss["benefit_per_order"].sum(),
                f"{loss_rate.iloc[0] * 100:.2f}% ({loss_rate.index[0]})",
            ],
        }
    )


# Multivariate analysis
def average_monthly_sales_top_products(data):
    required = ["product_name", "order_item_quantity", "ORDER_YEAR", "ORDER_MONTH"]
    if not _exists(data, required):
        return None

    top_products = data.groupby("product_name")["order_item_quantity"].sum().nlargest(10).index

    yearly_monthly_sales = (
        data[data["product_name"].isin(top_products)]
        .groupby(["product_name", "ORDER_YEAR", "ORDER_MONTH"], as_index=False)["order_item_quantity"]
        .sum()
    )

    monthly_avg_sales = (
        yearly_monthly_sales
        .groupby(["product_name", "ORDER_MONTH"], as_index=False)["order_item_quantity"]
        .mean()
    )

    fig = px.line(
        monthly_avg_sales,
        x="ORDER_MONTH",
        y="order_item_quantity",
        color="product_name",
        markers=True,
        title="Average Monthly Quantity Sold of Top Ten Products",
        labels={
            "ORDER_MONTH": "Month",
            "order_item_quantity": "Average quantity sold",
            "product_name": "Product",
        },
    )
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)
    fig.update_layout(height=600)

    return fig


def delay_rate_by_region_and_shipping_mode(data):
    required = ["order_region", "shipping_mode", "late_delivery_risk"]
    if not _exists(data, required):
        return None

    delay_rate = (
        data.groupby(["order_region", "shipping_mode"], as_index=False)["late_delivery_risk"]
        .mean()
    )
    delay_rate["late_delivery_pct"] = delay_rate["late_delivery_risk"] * 100

    fig = px.bar(
        delay_rate,
        x="order_region",
        y="late_delivery_pct",
        color="shipping_mode",
        barmode="group",
        title="Delay Rates of Different Shipping Modes in Various Regions",
        labels={
            "order_region": "Region",
            "late_delivery_pct": "Delay rate (%)",
            "shipping_mode": "Shipping mode",
        },
    )
    fig.update_xaxes(tickangle=45)
    fig.update_layout(height=560)
    fig.update_yaxes(range=[0, 100])

    return fig


def transportation_mode_distribution(data):
    required = ["shipping_date_dateorders", "market", "shipping_mode"]
    if not _exists(data, required):
        return None

    temp = data.copy()
    temp["shipping_date_dateorders"] = pd.to_datetime(temp["shipping_date_dateorders"], errors="coerce")
    temp["Year"] = temp["shipping_date_dateorders"].dt.year
    temp["Month"] = temp["shipping_date_dateorders"].dt.month

    shipping = (
        temp.groupby(["market", "shipping_mode"])
        .size()
        .reset_index(name="order_count")
    )

    fig = px.bar(
        shipping,
        x="market",
        y="order_count",
        color="shipping_mode",
        barmode="stack",
        title="Transportation Mode Distribution by Market",
        labels={
            "market": "Market",
            "order_count": "Order quantity",
            "shipping_mode": "Shipping mode",
        },
    )
    fig.update_layout(height=520)

    return fig


def late_delivery_heatmap_market_mode(data):
    required = ["market", "shipping_mode", "late_delivery_risk"]
    if not _exists(data, required):
        return None

    late_market_mode = (
        data.groupby(["market", "shipping_mode"], as_index=False)["late_delivery_risk"]
        .mean()
    )
    late_market_mode["late_rate_pct"] = late_market_mode["late_delivery_risk"] * 100

    pivot = late_market_mode.pivot(
        index="market",
        columns="shipping_mode",
        values="late_rate_pct",
    )

    if pivot.empty:
        return None

    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        title="Late Delivery Rate by Market and Shipping Mode",
        labels={
            "x": "Shipping mode",
            "y": "Market",
            "color": "Late delivery rate (%)",
        },
        zmin=0,
        zmax=100,
    )
    fig.update_layout(height=520)

    return fig


def category_sales_profit_late_risk(data):
    required = ["category_name", "sales", "benefit_per_order", "late_delivery_risk", "order_id"]
    if not _exists(data, required):
        return None

    category_summary = (
        data.groupby("category_name", as_index=False)
        .agg(
            total_sales=("sales", "sum"),
            avg_benefit=("benefit_per_order", "mean"),
            late_rate=("late_delivery_risk", "mean"),
            order_items=("order_id", "count"),
        )
    )

    category_summary = category_summary[category_summary["order_items"] >= 100].copy()
    if category_summary.empty:
        return None

    top_category_summary = (
        category_summary
        .sort_values("total_sales", ascending=False)
        .head(12)
        .copy()
    )
    top_category_summary["late_rate_pct"] = top_category_summary["late_rate"] * 100

    fig = px.scatter(
        top_category_summary,
        x="total_sales",
        y="avg_benefit",
        size="order_items",
        color="late_rate_pct",
        range_color=[0, 100],
        text="category_name",
        title="Top Product Categories: Sales, Profit, and Late Delivery Risk",
        labels={
            "total_sales": "Total sales",
            "avg_benefit": "Average benefit per order",
            "order_items": "Order items",
            "late_rate_pct": "Late delivery rate (%)",
            "category_name": "Product category",
        },
        hover_data=["category_name", "total_sales", "avg_benefit", "late_rate_pct", "order_items"],
    )
    fig.add_hline(y=0, line_dash="dash", annotation_text="Break-even")
    fig.update_traces(textposition="top center")
    fig.update_layout(height=620)

    return fig


# Dashboard render functions
def render_eda_report(data):

    st.header("Exploratory Data Analysis")
    # st.caption(
    #     "The EDA content follows the notebook, while the charts are displayed as "
    #     "interactive Plotly visualizations."
    # )

    with st.expander("Original team hypotheses", expanded=False):
        _show_table(original_hypotheses_table())

    tab1, tab2, tab3 = st.tabs([
        "Univariate Analysis",
        "Bivariate Analysis",
        "Multivariate Analysis",
    ])

    # 1. Univariate Analysis
    with tab1:
        _section("Univariate Analysis")

        _subsection("Sales Distribution and Outliers")
        _show_fig(sales_distribution(data), height=560)
        st.markdown(
            "**Interpretation:** The full sales range is retained and shown on a log-scaled x-axis so the concentration of typical orders and the high-value tail can be seen together. "
            "The median is marked with a dashed red line, and the histogram shows that most orders cluster in the lower-to-middle sales range while fewer high-value orders extend the right tail. "
            "These outliers are not removed because they represent real high-value business orders; keeping them preserves the authenticity of the dataset and avoids hiding important sales behavior."
        )
        with st.expander("Sales distribution statistics", expanded=False):
            _show_table(sales_distribution_stats(data), height=220)

        st.divider()

        _subsection("Benefit Per Order Distribution")
        _show_fig(profit_distribution(data), height=560)
        st.markdown(
            "**Interpretation:** The focused x-axis makes the negative-profit region easier to see. "
            "A meaningful share of order items falls below zero benefit, so product, region, shipping, and discount variables should be evaluated not only for sales but also for profitability risk."
        )
        with st.expander("Profit/loss statistic", expanded=False):
            _show_table(profit_distribution_stats(data), height=160)

        st.divider()

        _subsection("Top Products by Quantity Sold")
        _show_fig(top_products_by_sales(data, n=10), height=560)
        st.markdown(
            "**Interpretation:** A small number of products account for the highest quantities sold. "
            "This supports the original product-driven hypothesis: product identity and category are likely important for understanding sales performance."
        )

        st.divider()

        _subsection("Total Sales by Market")
        _show_fig(market_sales(data), height=520)
        st.markdown(
            "**Interpretation:** Total sales per customer are concentrated in certain markets. "
            "Market should therefore be included in later analysis because overall sales patterns may be shaped by geography."
        )

        st.divider()

        _subsection("Total Sales by Region")
        _show_fig(region_sales(data), height=680)
        st.markdown(
            "**Interpretation:** Sales also vary substantially by order region. "
            "This suggests that regional demand and market structure may influence sales outcomes, so regional comparisons should be rate-based rather than relying only on total counts."
        )

        st.divider()

        _subsection("Loss-Making Product Categories")
        _show_fig(loss_categories(data, n=10), height=560)
        st.markdown(
            "**Interpretation:** Loss-making orders are concentrated in particular product categories. "
            "This means product features do not only affect sales volume; they may also affect profitability risk."
        )

        st.divider()

        _subsection("Suspected Fraud by Region")
        _show_fig(fraud_by_region(data), height=680)
        st.markdown(
            "**Interpretation:** Suspected fraud orders are not evenly distributed across regions. "
            "However, this chart shows the distribution among fraud cases, so the report should also compare fraud rates by region before concluding that a region is truly high-risk."
        )

        st.divider()

        _subsection("Suspected Fraud by Payment Method")
        _show_fig(fraud_payment_distribution(data), height=500)
        st.markdown(
            "**Interpretation:** Suspected fraud cases are concentrated among payment types. "
            "This partially supports the team's hypothesis that payment method may be related to fraud risk."
        )

        st.divider()

        _subsection("Suspected Fraud by Product Category")
        _show_fig(fraud_category_distribution(data, n=10), height=620)
        st.markdown(
            "**Interpretation:** Some product categories appear more frequently in suspected fraud orders. "
            "This suggests that fraud risk may be associated with product type, although category-level fraud rates should also be checked because high-volume categories naturally produce more cases."
        )

        st.divider()

        _subsection("Top Suspected Fraud Customers")
        _show_fig(fraud_customer_distribution(data, n=10), height=620)
        st.markdown(
            "**Interpretation:** A few customer names appear more often among suspected fraud cases. "
            "This may indicate repeated suspicious activity, but customer-level interpretation should be handled carefully because names may not uniquely identify individuals."
        )

        st.divider()

        _subsection("RFM Customer Segmentation")
        rfm_df = create_rfm_table(data)

        if rfm_df is None or rfm_df.empty:
            st.info("RFM analysis requires customer_id, order_date_dateorders, and sales columns.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Customers", f"{rfm_df['customer_id'].nunique():,}")
            c2.metric("Average recency", f"{rfm_df['recency'].mean():.1f} days")
            c3.metric("Average frequency", f"{rfm_df['frequency'].mean():.2f}")
            c4.metric("Average monetary", _format_money(rfm_df["monetary"].mean()))

            _show_fig(rfm_value_pyramid(rfm_df), height=620)
            st.markdown(
                "**Interpretation:** The RFM value pyramid shows that customers are distributed unevenly across value groups. "
                "This supports using RFM stratification as a descriptive tool and potentially as a feature source in later modelling."
            )
            st.divider()
            _show_fig(rfm_segment_chart(rfm_df), height=620)
            st.markdown(
                "**Interpretation:** The pie chart gives a second view of the RFM segment distribution. "
                "It confirms that customer value groups are imbalanced, so recommendations should not assume that all customer segments contribute equally."
            )

            with st.expander("View RFM table", expanded=False):
                _show_table(rfm_df.head(100), height=420)

    # 2. Bivariate Analysis
    with tab2:
        _section("Bivariate Analysis")

        _subsection("Monthly Sales by Year")
        _show_fig(monthly_sales_trend(data), height=580)
        st.markdown(
            "**Interpretation:** Monthly sales vary across years and months rather than staying flat. "
            "This supports the team's expectation that sales may have a time pattern, although formal seasonality would need additional testing in the modelling stage."
        )

        st.divider()

        _subsection("Order Volume by Hour of Day")
        _show_fig(hourly_order_trend(data), height=540)
        st.markdown(
            "**Interpretation:** Order volume changes by hour of day, showing that customer activity is not evenly distributed across the 24-hour cycle. "
            "Hourly timing may be useful as a feature if the final model predicts demand, delivery risk, or operational load."
        )

        st.divider()

        _subsection("Late Delivery Rate by Region")
        _show_fig(late_delivery_by_region(data), height=660)
        st.markdown(
            "**Interpretation:** Late delivery rates differ by region. "
            "This introduces an important operational risk pattern that was not the main focus of the original hypotheses but appears strong enough to consider for modelling."
        )

        st.divider()

        _subsection("Late Delivery Risk by Shipping Mode and Market")
        _show_fig(late_delivery_risk_by_mode_and_market(data), height=600)
        st.markdown(
            "**Interpretation:** This bivariate analysis shows that late delivery risk changes by both shipping mode and market. "
            "The finding adds an operational-risk dimension to the original hypotheses, which focused more on product features and fraud."
        )

        st.divider()

        _subsection("Sales vs Benefit Per Order")
        corr = sales_benefit_correlation(data)
        if corr is not None:
            st.metric("Correlation between sales and benefit per order", f"{corr:.4f}")
        _show_fig(sales_benefit_scatter(data), height=640)
        st.markdown(
            "**Interpretation:** Sales and benefit are related, but high sales do not guarantee positive profit. "
            "This means the team should be careful about defining success only as higher sales; profitability and loss risk are separate business concerns."
        )

        st.divider()

        _subsection("Loss Rate by Region")
        _show_fig(loss_rate_by_region(data), height=660)
        st.markdown(
            "**Interpretation:** Loss rates differ across regions, which suggests that regional business conditions, product mix, discounting, or logistics may influence profitability. "
            "This is a risk pattern worth checking before modelling."
        )
        with st.expander("Loss-rate summary statistics", expanded=False):
            _show_table(loss_rate_by_region_stats(data), height=220)

    # 3. Multivariate Analysis
    with tab3:
        _section("Multivariate Analysis")

        _subsection("Average Monthly Quantity for Top Products")
        _show_fig(average_monthly_sales_top_products(data), height=660)
        st.markdown(
            "**Interpretation:** The top products have different monthly quantity patterns, so product performance is not uniform over time. "
            "This strengthens the idea that both product features and time features should be considered together."
        )

        st.divider()

        _subsection("Late Delivery Rate by Region and Shipping Mode")
        _show_fig(delay_rate_by_region_and_shipping_mode(data), height=660)
        st.markdown(
            "**Interpretation:** Late delivery risk changes across both region and shipping mode. "
            "This is already a multivariate pattern because the relationship between delivery risk and shipping mode depends on region."
        )

        st.divider()

        _subsection("Shipping Mode Mix by Market")
        _show_fig(transportation_mode_distribution(data), height=580)
        st.markdown(
            "**Interpretation:** Although the original title says monthly distribution, the code groups orders by market and shipping mode. "
            "The chart still shows useful information: markets rely on different mixes of shipping modes, which may partly explain market-level delivery risk."
        )

        st.divider()

        _subsection("Late Delivery Rate by Market and Shipping Mode")
        _show_fig(late_delivery_heatmap_market_mode(data), height=580)
        st.markdown(
            "**Interpretation:** This multivariate heatmap compares late delivery risk across market and shipping mode at the same time. "
            "The risk level depends on the combination of geography and shipping choice, so later models should allow interactions between these features."
        )

        st.divider()

        _subsection("Category Sales, Profit, and Late Delivery Risk")
        _show_fig(category_sales_profit_late_risk(data), height=680)
        st.markdown(
            "**Interpretation:** This multivariate product-category chart avoids the label overlap problem by using offset annotations. "
            "It shows that product categories differ across sales, average profit, order volume, and late delivery risk, so product features should be combined with operational variables in later modelling."
        )

    st.divider()


def render_fraud_and_delivery_risk(data):
    st.header("Fraud and Delivery Risk Analysis")
    # st.caption("This page uses the same fraud and delay-related content as the EDA notebook.")

    tab1, tab2 = st.tabs(["Fraud risk", "Delivery risk"])

    with tab1:
        _subsection("Suspected Fraud by Region")
        _show_fig(fraud_by_region(data), height=700)
        st.markdown(
            "**Interpretation:** Suspected fraud orders are not evenly distributed across regions. "
            "However, this chart shows the distribution among fraud cases, so the report should also compare fraud rates by region before concluding that a region is truly high-risk."
        )

        st.divider()

        _subsection("Suspected Fraud by Payment Method")
        _show_fig(fraud_payment_distribution(data), height=520)
        st.markdown(
            "**Interpretation:** Suspected fraud cases are concentrated among payment types. "
            "This partially supports the team's hypothesis that payment method may be related to fraud risk."
        )

        st.divider()

        _subsection("Suspected Fraud by Product Category")
        _show_fig(fraud_category_distribution(data, n=10), height=640)
        st.markdown(
            "**Interpretation:** Some product categories appear more frequently in suspected fraud orders. "
            "This suggests that fraud risk may be associated with product type, although category-level fraud rates should also be checked because high-volume categories naturally produce more cases."
        )

        st.divider()

        _subsection("Top Suspected Fraud Customers")
        _show_fig(fraud_customer_distribution(data, n=10), height=640)
        st.markdown(
            "**Interpretation:** A few customer names appear more often among suspected fraud cases. "
            "This may indicate repeated suspicious activity, but customer-level interpretation should be handled carefully because names may not uniquely identify individuals."
        )

    with tab2:
        _subsection("Late Delivery Rate by Region")
        _show_fig(late_delivery_by_region(data), height=680)
        st.markdown(
            "**Interpretation:** Late delivery rates differ by region. "
            "This introduces an important operational risk pattern that was not the main focus of the original hypotheses but appears strong enough to consider for modelling."
        )

        st.divider()

        _subsection("Late Delivery Risk by Shipping Mode and Market")
        _show_fig(late_delivery_risk_by_mode_and_market(data), height=620)
        st.markdown(
            "**Interpretation:** This bivariate analysis shows that late delivery risk changes by both shipping mode and market. "
            "The finding adds an operational-risk dimension to the original hypotheses, which focused more on product features and fraud."
        )

        st.divider()

        _subsection("Delay Rate by Region and Shipping Mode")
        _show_fig(delay_rate_by_region_and_shipping_mode(data), height=680)
        st.markdown(
            "**Interpretation:** Late delivery risk changes across both region and shipping mode. "
            "This is already a multivariate pattern because the relationship between delivery risk and shipping mode depends on region."
        )

        st.divider()

        _subsection("Late Delivery Rate by Market and Shipping Mode")
        _show_fig(late_delivery_heatmap_market_mode(data), height=600)
        st.markdown(
            "**Interpretation:** This multivariate heatmap compares late delivery risk across market and shipping mode at the same time. "
            "The risk level depends on the combination of geography and shipping choice, so later models should allow interactions between these features."
        )


def render_rfm_report(data):
    rfm_df = create_rfm_table(data)

    if rfm_df is None or rfm_df.empty:
        st.info("RFM analysis requires customer_id, order_date_dateorders, and sales columns.")
        return

    st.header("RFM Customer Segmentation")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{rfm_df['customer_id'].nunique():,}")
    c2.metric("Average recency", f"{rfm_df['recency'].mean():.1f} days")
    c3.metric("Average frequency", f"{rfm_df['frequency'].mean():.2f}")
    c4.metric("Average monetary", _format_money(rfm_df["monetary"].mean()))

    st.divider()

    _subsection("Customer Value Pyramid")
    _show_fig(rfm_value_pyramid(rfm_df), height=640)
    st.markdown(
        "**Interpretation:** The RFM value pyramid shows that customers are distributed unevenly across value groups. "
        "This supports using RFM stratification as a descriptive tool and potentially as a feature source in later modelling."
    )

    st.divider()

    _subsection("Customer Segmentation Distribution")
    _show_fig(rfm_segment_chart(rfm_df), height=640)
    st.markdown(
        "**Interpretation:** The pie chart gives a second view of the RFM segment distribution. "
        "It confirms that customer value groups are imbalanced, so recommendations should not assume that all customer segments contribute equally."
    )

    with st.expander("View RFM table", expanded=False):
        _show_table(rfm_df.head(100), height=420)

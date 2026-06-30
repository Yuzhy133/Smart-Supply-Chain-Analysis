import warnings
import logging

import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


MODELING_OVERVIEW_ANALYSIS = """
This dashboard implemented an end-to-end supply chain data analysis and modeling workflow, encompassing feature engineering, training, and comparing multiple regression models (linear regression, random forest, gradient boosting) as well as a time series model (Prophet). Based on the models, sales forecasting performance and business insights were ultimately provided.
""".strip()


MODEL_SELECTION_ANALYSIS = """
## Model Selection & Justification

| Model | Rationale | Applicable Scenario |
|-------|-----------|---------------------|
| **Linear Regression** | Baseline model, simple and interpretable, fast computation | Features have approximately linear relationship with target |
| **Random Forest** | Captures non-linear relationships, robust to outliers, provides feature importance | Tabular data, complex feature interactions |
| **Gradient Boosting** | Higher accuracy, progressively optimizes residuals | High accuracy requirements |
| **Prophet** | Handles missing values automatically, supports multiple seasonalities | Clear time in sales |

""".strip()


def prepare_sales_features(data):
    data = data.copy()

    if "ORDER_MONTH" not in data.columns and "order_date_dateorders" in data.columns:
        data["ORDER_MONTH"] = data["order_date_dateorders"].dt.month

    if "ORDER_YEAR" not in data.columns and "order_date_dateorders" in data.columns:
        data["ORDER_YEAR"] = data["order_date_dateorders"].dt.year

    feature_cols = [
        "category_id",
        "customer_city",
        "customer_country",
        "market",
        "order_city",
        "order_country",
        "order_item_quantity",
        "order_region",
        "product_name",
        "ORDER_MONTH",
        "ORDER_YEAR",
    ]

    available_features = [col for col in feature_cols if col in data.columns]

    if "sales" not in data.columns:
        raise ValueError("The target column `sales` is not available in the dataset.")

    if len(available_features) == 0:
        raise ValueError("None of the notebook modeling feature columns are available.")

    model_data = data[available_features + ["sales"]].dropna().copy()

    sales_features = model_data[available_features]
    target = model_data["sales"]

    return sales_features, target, available_features


def encode_and_scale_features(sales_features):
    sales_features = sales_features.copy()

    categorical_cols = sales_features.select_dtypes(include=["object"]).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        sales_features[col] = le.fit_transform(sales_features[col].astype(str))
        label_encoders[col] = le

    scaler = StandardScaler()
    sales_features_scaled = scaler.fit_transform(sales_features)

    return sales_features, sales_features_scaled, label_encoders, scaler


# MODEL_TUNING_ANALYSIS = """
# ## Model Tuning
#
# Random forest adds a parameter of max_features=0.3, which limits the number of features to force differences between trees and avoid all trees relying on the same strong features.
# """.strip()


def get_models(random_state=42, use_max_features=False):

    if use_max_features:
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_features=0.3,
            random_state=random_state,
        )
    else:
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
        )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": rf_model,
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100,
            random_state=random_state,
        ),
    }

    return models


MODEL_TRAINING_EVALUATION_ANALYSIS = """
## Model Training & Evaluation

- **MSE (Mean Squared Error)**: Measures deviation between predicted and actual values; lower is better
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model; closer to 1 is better
""".strip()


def train_and_evaluate_models(models, X_train, X_test, y_train, y_test):
    results = {}
    predictions = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results[name] = {
            "MSE": mse,
            "R2": r2,
            "MAE": mae,
            "RMSE": np.sqrt(mse),
        }

        predictions[name] = pd.DataFrame(
            {
                "Actual": np.asarray(y_test),
                "Predicted": y_pred,
            }
        )

    results_df = (
        pd.DataFrame(results)
        .T
        .reset_index()
        .rename(columns={"index": "Model"})
        .sort_values("R2", ascending=False)
        .reset_index(drop=True)
    )

    return results_df, predictions


# UNIQUE_FEATURE_SPLIT_ANALYSIS = """
# ## Select Unique Feature Combinations
#
# After model tuning, it is essential to ensure that the model's performance is not artificially inflated due to **feature combination leakage**. This occurs when identical feature combinations appear in both the training and test sets, allowing the model to "memorize" outcomes rather than learning general patterns.
#
# The function `get_unique_feature_split()` addresses this issue by ensuring that the training and test sets contain **completely independent feature combinations**.
#
# ---
#
# | Step | Description |
# |------|-------------|
# | 1 | Convert the feature array back to a DataFrame with original column names |
# | 2 | Identify all unique feature combinations using `drop_duplicates()` |
# | 3 | For each unique combination, randomly select **one representative sample** |
# | 4 | Split these representative samples into training and test sets (80/20) |
# | 5 | Train models on the unique-feature training set and evaluate on the unique-feature test set |
#
# **Key Insight:** This approach guarantees that no feature combination appears in both training and test sets, providing a more rigorous evaluation of model generalization.
# """.strip()


def get_unique_feature_split(features, target, feature_names, test_size=0.2, random_state=42):
    features_df = pd.DataFrame(features, columns=feature_names)

    unique_combinations = features_df.drop_duplicates()

    repetition_rate = (1 - len(unique_combinations) / len(features_df)) * 100

    representative_indices = []

    for _, group in features_df.groupby(features_df.columns.tolist()):
        representative_indices.append(
            group.sample(1, random_state=random_state).index[0]
        )

    unique_features = features[representative_indices]

    if hasattr(target, "iloc"):
        unique_target = target.iloc[representative_indices]
    else:
        unique_target = target[representative_indices]

    X_train_unique, X_test_unique, y_train_unique, y_test_unique = train_test_split(
        unique_features,
        unique_target,
        test_size=test_size,
        random_state=random_state,
    )

    split_info = {
        "Number of original samples": len(features_df),
        "Number of unique feature combinations": len(unique_combinations),
        "Repetition rate": repetition_rate,
        "Unique training set size": len(X_train_unique),
        "Unique test set size": len(X_test_unique),
    }

    return X_train_unique, X_test_unique, y_train_unique, y_test_unique, split_info


# OVERFITTING_ANALYSIS = """
# ## Overfitting Test
#
# **Conclusion**: Training and test R² are nearly identical. The model shows no significant overfitting and generalizes well.
# """.strip()


def check_overfitting(model, X_train, X_test, y_train, y_test):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    difference = train_score - test_score

    if difference > 0.1:
        judgment = "Severe overfitting"
    elif difference > 0.05:
        judgment = "Possibly overfitting"
    else:
        judgment = "Goodness of fit is normal"

    return {
        "Train set R2": train_score,
        "Test set R2": test_score,
        "Difference": difference,
        "Judgment": judgment,
    }


MODEL_PERFORMANCE_CONCLUSION = """
**Conclusion**: Random Forest performs best on this task with R² = 0.9952 and lowest MSE.
""".strip()


def train_sales_models(data, sample_size=None, test_size=0.2, random_state=42):
    sales_features, target, feature_names = prepare_sales_features(data)

    if sample_size is not None and len(sales_features) > sample_size:
        sample_index = sales_features.sample(sample_size, random_state=random_state).index
        sales_features = sales_features.loc[sample_index]
        target = target.loc[sample_index]

    encoded_sales_features, sales_features_scaled, label_encoders, scaler = encode_and_scale_features(
        sales_features
    )

    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
        sales_features_scaled,
        target,
        test_size=test_size,
        random_state=random_state,
    )

    original_models = get_models(random_state=random_state, use_max_features=False)

    original_results, original_predictions = train_and_evaluate_models(
        original_models,
        X_train_orig,
        X_test_orig,
        y_train_orig,
        y_test_orig,
    )

    X_train, X_test, y_train, y_test, split_info = get_unique_feature_split(
        sales_features_scaled,
        target,
        feature_names=encoded_sales_features.columns,
        test_size=test_size,
        random_state=random_state,
    )

    models = get_models(random_state=random_state, use_max_features=False)

    results_df, predictions = train_and_evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    rf_original = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
    )
    rf_original.fit(X_train_orig, y_train_orig)
    original_r2 = rf_original.score(X_test_orig, y_test_orig)

    rf_unique = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
    )
    rf_unique.fit(X_train, y_train)
    unique_r2 = rf_unique.score(X_test, y_test)

    comparison = {
        "Original Classification Random Forest R2": original_r2,
        "Unique feature classification Random Forest R2": unique_r2,
        "R2 difference": unique_r2 - original_r2,
    }

    overfitting_result = check_overfitting(
        models["Random Forest"],
        X_train,
        X_test,
        y_train,
        y_test,
    )

    rf_model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
    )
    rf_model.fit(X_train, y_train)

    feature_importance = (
        pd.DataFrame(
            {
                "feature": encoded_sales_features.columns,
                "importance": rf_model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "results": results_df,
        "predictions": predictions,
        "pipelines": models,
        "features": list(encoded_sales_features.columns),
        "target": "sales",
        "label_encoders": label_encoders,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "original_results": original_results,
        "original_predictions": original_predictions,
        "split_info": split_info,
        "comparison": comparison,
        "overfitting_result": overfitting_result,
        "feature_importance": feature_importance,
    }


def model_performance_chart(results_df):
    if results_df is None or results_df.empty:
        return None

    fig = px.bar(
        results_df.sort_values("R2"),
        x="R2",
        y="Model",
        orientation="h",
        title="Model Performance Comparison by R²",
    )

    fig.update_layout(
        xaxis_title="R² Score",
        yaxis_title="Model",
    )

    return fig


def prediction_scatter(prediction_df, model_name):
    if prediction_df is None or prediction_df.empty:
        return None

    plot_df = prediction_df.copy()

    if len(plot_df) > 3000:
        plot_df = plot_df.sample(3000, random_state=42)

    fig = px.scatter(
        plot_df,
        x="Actual",
        y="Predicted",
        title=f"Actual vs Predicted Sales ({model_name})",
        opacity=0.6,
    )

    min_value = min(plot_df["Actual"].min(), plot_df["Predicted"].min())
    max_value = max(plot_df["Actual"].max(), plot_df["Predicted"].max())

    fig.add_shape(
        type="line",
        x0=min_value,
        y0=min_value,
        x1=max_value,
        y1=max_value,
        line=dict(color="red", dash="dash"),
    )

    fig.update_layout(
        xaxis_title="Actual Sales",
        yaxis_title="Predicted Sales",
    )

    return fig


# FEATURE_IMPORTANCE_ANALYSIS = """
# ## Feature Importance Analysis (Random Forest)
#
# **Key Observations:**
#
# - **Product Name** is the most influential feature, with an importance score of **0.446**. This indicates that the specific product being sold explains nearly 45% of the variance in sales, suggesting significant sales differences across individual products.
#
# - **Order Item Quantity** ranks second with a score of **0.277**. Purchase quantity directly affects total sales value, which aligns with business intuition.
#
# - **Category ID** follows closely at **0.271**, showing that product categories also play a substantial role in determining sales performance.
#
# - **Time-related features** (`ORDER_YEAR` and `ORDER_MONTH`) have very low importance scores (0.003 each), indicating that sales are primarily driven by product characteristics rather than temporal factors.
#
# - **Geographical features** (`order_country`, `order_city`, `customer_city`, `order_region`, `customer_country`, `market`) show near-zero importance, suggesting that location has minimal impact on sales in this dataset.
#
# **Business Implications:**
#
# 1. **Focus on product-level strategies**: Inventory management, pricing, and promotions should be tailored to individual products rather than broad categories or regions.
#
# 2. **Monitor order quantities**: Bundling strategies or volume discounts could further leverage the impact of order quantity on sales.
#
# 3. **Category matters but less than specific products**: While category is important, product-level differentiation provides stronger predictive power.
#
# 4. **Time and location are secondary**: Sales forecasting models can prioritize product-related features without significant loss of accuracy from ignoring temporal or geographical variations.
#
# **Conclusion:** The Random Forest model effectively identifies that product identity (name) and purchase quantity are the dominant drivers of sales, while most other features contribute minimally to prediction accuracy.
# """.strip()


def feature_importance_table(model_result, model_name=None):
    if model_result is None:
        return None

    if "feature_importance" not in model_result:
        return None

    return model_result["feature_importance"]


def feature_importance_chart(importance_df, n=15):
    if importance_df is None or importance_df.empty:
        return None

    top_importance = importance_df.head(n).sort_values("importance")

    fig = px.bar(
        top_importance,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Top {n} Random Forest Feature Importance Ranking",
    )

    fig.update_layout(
        xaxis_title="Feature Importance",
        yaxis_title="Features",
    )

    return fig


TIME_SERIES_ANALYSIS = """
## Time Series Analysis
""".strip()


def prophet_forecast(data, periods=30):
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError(
            "Prophet is not installed. Please install it with: pip install prophet"
        )

    if "order_date_dateorders" not in data.columns or "sales" not in data.columns:
        raise ValueError(
            "Prophet forecasting requires `order_date_dateorders` and `sales` columns."
        )

    daily_sales = (
        data.groupby(data["order_date_dateorders"].dt.date)["sales"]
        .sum()
        .reset_index()
    )

    daily_sales.columns = ["ds", "y"]
    daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )

    model.fit(daily_sales)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    actual = daily_sales["y"].values
    predicted = forecast["yhat"].head(len(actual)).values

    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    prophet_results = {
        "MAE": mae,
        "MAPE": mape,
    }

    return model, forecast, prophet_results


def analyze_prophet_components(prophet_model, prophet_forecast):
    main_components = ["trend", "yearly", "weekly"]
    component_data = []

    for col in main_components:
        if col in prophet_forecast.columns:
            importance = prophet_forecast[col].std()
            component_data.append((col, importance))

    component_data.sort(key=lambda x: x[1], reverse=True)

    component_info = {
        "trend": "Long-term trend",
        "yearly": "Annual seasonality",
        "weekly": "Weekly seasonality",
        "additive_terms": "Total additional impact",
        "multiplicative_terms": "Total factor productivity effect",
    }

    component_df = pd.DataFrame(
        [
            {
                "Component Name": col,
                "Standard deviation": std,
                "Explanation": component_info.get(col, "Other components"),
            }
            for col, std in component_data
        ]
    )

    config_info = []

    for seasonality, config in prophet_model.seasonalities.items():
        config_info.append(
            {
                "Seasonality": seasonality,
                "Period": config["period"],
                "Fourier order": config["fourier_order"],
            }
        )

    config_df = pd.DataFrame(config_info)

    return component_df, config_df

def prophet_forecast(data):
    try:
        from prophet import Prophet
    except ImportError:
        raise ImportError(
            "Prophet is not installed. Please install it with: pip install prophet"
        )

    if "order_date_dateorders" not in data.columns:
        raise ValueError("Column `order_date_dateorders` is required for Prophet forecasting.")

    if "sales" not in data.columns:
        raise ValueError("Column `sales` is required for Prophet forecasting.")

    # Prepare data
    daily_sales = (
        data.groupby(data["order_date_dateorders"].dt.date)["sales"]
        .sum()
        .reset_index()
    )

    daily_sales.columns = ["ds", "y"]
    daily_sales["ds"] = pd.to_datetime(daily_sales["ds"])

    # Create and train a Prophet model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )

    model.fit(daily_sales)

    # Create future dataframe
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Evaluation model
    from sklearn.metrics import mean_absolute_error

    actual = daily_sales["y"].values
    predicted = forecast["yhat"].head(len(actual)).values

    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    prophet_results = {
        "MAE": mae,
        "MAPE": mape,
    }

    return model, forecast, daily_sales, prophet_results


def prophet_forecast_chart(forecast, daily_sales):
    if forecast is None or daily_sales is None:
        return None

    fig = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="Sales Forecast with Prophet",
    )

    fig.add_scatter(
        x=daily_sales["ds"],
        y=daily_sales["y"],
        mode="markers",
        name="Actual Sales",
    )

    fig.add_scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        mode="lines",
        name="Upper Bound",
        line=dict(width=0),
        showlegend=False,
    )

    fig.add_scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        mode="lines",
        name="Lower Bound",
        fill="tonexty",
        line=dict(width=0),
        showlegend=True,
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sales",
    )

    fig.update_yaxes(rangemode="tozero")

    return fig


PROPHET_FORECAST_ANALYSIS = """
The chart above shows the Prophet model's sales forecast from January 2015 to September 2017.

**Key Observations:**

- **Dots**: Actual historical sales data
- **Lines**: Predicted sales values
- **Blue shading**: 95% confidence interval

**Findings:**

1. The model captures both the **overall upward trend** and **seasonal patterns** in sales.
2. Actual sales generally fall within the confidence interval, indicating a good fit.
3. Sales show clear **periodic peaks** suggesting regular seasonal demand cycles.

**Business Implications:**

- The model can be used for **short-term sales planning**.
- Seasonal patterns help with **inventory preparation** ahead of peak periods.
""".strip()


def analyze_prophet_components(prophet_model, prophet_forecast):
    if prophet_model is None or prophet_forecast is None:
        return None, None

    # 1. Model configuration information
    config_info = []

    for seasonality, config in prophet_model.seasonalities.items():
        config_info.append(
            {
                "Seasonality": seasonality,
                "Period": config["period"],
                "Fourier order": config["fourier_order"],
            }
        )

    config_df = pd.DataFrame(config_info)

    # 2. Component impact analysis
    main_components = ["trend", "yearly", "weekly"]
    component_data = []

    for col in main_components:
        if col in prophet_forecast.columns:
            importance = prophet_forecast[col].std()
            component_data.append((col, importance))

    component_data.sort(key=lambda x: x[1], reverse=True)

    component_info = {
        "trend": "Long-term trend",
        "yearly": "Annual seasonality",
        "weekly": "Weekly seasonality",
        "additive_terms": "Total additional impact",
        "multiplicative_terms": "Total factor productivity effect",
    }

    component_df = pd.DataFrame(
        [
            {
                "Component Name": col,
                "Standard deviation": std,
                "Explanation": component_info.get(col, "Other components"),
            }
            for col, std in component_data
        ]
    )

    return component_df, config_df


def prophet_component_charts(prophet_model, forecast):
    if prophet_model is None or forecast is None:
        return None

    fig = prophet_model.plot_components(forecast)

    return fig


PROPHET_COMPONENT_ANALYSIS = """
**Key Findings:**

- **Trend** is the dominant factor, accounting for nearly 1.9× the impact of seasonality. This indicates a strong long-term sales growth trend.

- **Yearly seasonality** shows clear annual cycles, with peaks around November and troughs in January/July.

- **Weekly seasonality** reflects weekend effects — sales are higher on Saturdays and Fridays, lower on Wednesdays.

**Business Insights:**

1. **Focus on long-term growth strategies** — trend drives sales more than seasonal patterns.
2. **Prepare inventory for annual peaks** — especially before November.
3. **Adjust staffing for weekends** — higher sales expected on Fridays and Saturdays.
""".strip()
#
#
# MODEL_LIMITATIONS_ANALYSIS = """
# ## Model Limitations & Potential Failure Modes
#
# ### Linear Regression
# - **Limitation**: Assumes linear relationships between features and target.
# - **Failure mode**: Poor performance when real relationships are nonlinear (e.g., seasonal spikes in sales).
#
# ### Random Forest
# - **Limitation**: Can overfit if not tuned properly (though it performed well here).
# - **Failure mode**: May fail to extrapolate beyond historical data range; not suitable for long-term forecasting.
#
# ### Gradient Boosting
# - **Limitation**: Sensitive to noisy data and outliers.
# - **Failure mode**: Performance degrades if new data has different distribution (concept drift).
#
# ### Prophet (Time Series)
# - **Limitation**: Assumes seasonal patterns repeat regularly.
# - **Failure mode**: Fails during unexpected events (e.g., supply chain disruption, holidays not in training data).
#
# ### General Limitations
# - Data quality issues (missing values in `order_zipcode`, `product_description`).
# - Feature engineering may not capture complex interactions between product, customer, and time.
# - Models were trained on historical data up to 2017; may not generalize to recent market changes.
# """.strip()

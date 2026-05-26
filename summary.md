# Data Preprocessing Summary

## 1. Dataset Description
### 1.1. Dataset Source

The dataset used in this project is obtained from the official Mendeley Data repository published by DataCo Company:

Dataset Link: https://data.mendeley.com/datasets/8gx2fvg2k6/5

The dataset is a publicly available smart supply chain dataset that simulates real-world enterprise supply chain operations. It covers the complete business process from order generation, customer purchases, inventory circulation, logistics transportation, and final delivery.

This dataset is widely used in research areas such as:

- Supply chain management
- Sales forecasting
- Customer behavior analysis
- Fraud detection and risk control
- Logistics optimization
- Business intelligence and data mining

Because the dataset contains realistic commercial operational records, it is highly suitable for the research topic of this project: *Smart Supply Chain Analysis and Prediction Improvement*.

The dataset is publicly accessible through the Mendeley Data platform. You can access it by **following steps**:


1. Visit the official dataset website.
2. Register or log in to the Mendeley Data platform.
3. Download the CSV dataset file.
4. Use Python or other analytical tools for further processing.

The dataset is free to access and provides good reproducibility for academic research.

### 1.2. Dataset Size and Format
The dataset is stored in **CSV (Comma-Separated Values)** format, which is a standard structured tabular data format.

Based on the actual dataset file analysis, the dataset contains:

- **180,519 transaction records (rows)**
- **53 feature variables (columns)**

This is considered a relatively large-scale structured business dataset.

The dataset provides sufficient data volume for:

- Machine learning model training
- Time-series forecasting
- Customer segmentation
- Fraud detection
- Supply chain optimization analysis

### 1.3. Data Type Distribution

| Data Type | Quantity | Included Fields |
|---|---|---|
| Object/String | 24 | Order Status, Order Region, Order Country, Order City, Market, Type, Category Name, Department Name, Product Name, Customer Segment, Customer Country, Customer State, Customer City, Customer Street, Customer Fname, Customer Lname, Customer Email, Shipping Mode, Delivery Status, Order Date, Shipping Date, Customer Password, Product Image, Product Description |
| Float64 | 15 | Sales, Benefit per order, Order Profit Per Order, Order Item Product Price, Order Item Discount, Order Item Discount Rate, Order Item Profit Ratio, Order Item Total, Product Price, Sales per customer, Latitude, Longitude, Order Item Cardprod Id, Order Item Id, Product Card Id |
| Int64 | 14 | Order Id, Customer Id, Department Id, Category Id, Order Item Quantity, Days for shipping (real), Days for shipment (scheduled), Late_delivery_risk, Product Status, Order Customer Id, Order Item Product Price, Order Item Discount, Order Item Quantity, Product Category Id |



### 1.4. Feature Types
The dataset contains different data formats and feature types, including:

| Data Category	| Included Fields|
|---|---|
|Sales and Profit Data	|Sales, Benefit per order, Order Profit Per Order, Order Item Product Price, Order Item Discount, Order Item Discount Rate, Order Item Profit Ratio, Order Item Total, Sales per customer, Product Price|
|Order Status Data	|Order Status, Delivery Status, Late_delivery_risk, Days for shipping (real), Days for shipment (scheduled), Shipping Mode|
|Customer Behavior Data	|Customer Id, Customer Segment, Customer Country, Customer State, Customer City, Customer Street, Customer Fname, Customer Lname, Customer Email|
|Time-Series Data	|Order Date, Shipping Date|
|Geographic Data	|Latitude, Longitude, Order Region, Order Country, Order City, Market
|Product Classification Data	|Category Name, Department Name, Product Name, Product Category Id, Product Card Id, Product Description, Product Image, Product Status|

---

## 2. Data Processing Tool

In this project, the dataset will mainly be read and processed using Python. The primary tools and libraries include:

| Tool | Functions |
|---|---|
| Pandas | Used for reading CSV files, data cleaning, missing value processing, duplicate value detection, outlier detection, field selection, field renaming, data merging, grouped statistical analysis, time feature conversion, pivot table analysis, and DataFrame-based tabular data management. |
| NumPy | Used for efficient numerical computation, matrix operations, multidimensional array processing, mathematical calculations, random number generation, vectorized operations, and low-level scientific computing support. |
| Scikit-learn | Used for machine learning modeling and evaluation, including logistic regression, linear regression, decision trees, random forests, K-Means clustering, data standardization, feature engineering, train-test splitting, cross-validation, and model performance evaluation. |
| Matplotlib | Used for creating basic data visualization charts, including line charts, bar charts, scatter plots, pie charts, histograms, time-series trend charts, and prediction result visualizations. |
| Seaborn | Used for advanced statistical visualization, including heatmaps, box plots, violin plots, distribution plots, categorical statistical plots, correlation analysis plots, and regression relationship visualizations. |

---

## 3. Data Quality

### 3.1 Missing Rate
Through systematic preprocessing, **the missing rate in the final analysis dataset has been reduced to 0%.** The process was as follows:

*   **Initial Missing Value Identification**: The original data contained numerous missing values in columns such as `customer_zipcode`, `order_zipcode`, and `product_description`. The `product_description` column was almost entirely missing, and `order_zipcode` had an extremely high missing rate.
*   **Handling Rationale & Operations**:
    1.  **Deletion of High-Missing Features**: Based on the principle that "imputing features missing beyond a reasonable threshold may introduce greater bias," columns with low analytical value and extremely high missing rates were directly deleted. These included `product_description`, `order_zipcode`, `customer_zipcode`, `customer_email`, and `customer_password`.
    2.  **Data Filtering**: By setting `cutoff_date = pd.Timestamp('2017-10-01')`, only data before October 1, 2017, was retained. This ensured data completeness within the analysis period and may have filtered out incomplete subsequent records.

**Conclusion**: After the above cleaning, a check via `data.isnull().sum()` on the core dataset used for modeling and analysis shows consistent non-null counts for all columns, indicating **no missing values**, providing a stable foundation for subsequent analysis.

### 3.2 Class Balance
The distribution of categories across different dimensions in the dataset reflects the inherent imbalances of real-world business operations. 

*   **Target Variable (`late_delivery_risk`)**: The mean is 0.55, indicating a roughly balanced, slightly risk-leaning distribution between "at-risk" (1) and "not at-risk" (0) orders. This distribution is favorable for training stable binary classification models.
*   **Key Categorical Features**:
    *   **Geographic Distribution**: Sales are highly concentrated in key markets like `US` and `LATAM`, and regions like `West of USA` and `East of USA`, consistent with the company's market focus.
    *   **Product Distribution**: Sales and losses are concentrated in a few top categories like `Cleats`, `Indoor/Outdoor Games`, and `Cardio Equipment`, aligning with the retail "long tail" effect.
    *   **Risk Events**: Fraudulent orders (`SUSPECTED_FRAUD`) and specific payment types (`TRANSFER`) constitute a minority of overall orders. However, targeted analysis revealed significant patterns within them.

**Summary**: The class imbalance in the dataset is a true reflection of business reality and does not constitute data bias. For subsequent in-depth analysis of minority classes, techniques like weighting can be employed to ensure model sensitivity.

### 3.3 Outlier Handling Method
This project adopted a strategy of "identify, analyze, and handle within business context" for outliers, rather than simple removal.

*   **Identification Methods**:
    *   Descriptive statistics via `.describe()` identified potential outliers, such as extreme negative values in `benefit_per_order`  and very high values in `sales_per_customer`.
    *   Business logic was used to define "anomalies". For example, categorizing all orders with `benefit_per_order < 0` as "loss orders" for analysis.
*   **Handling Strategy**:
    *   **Business-context Retention**: Statistical outliers (e.g. high losses, very high sales) were not truncated or deleted. Instead, they were treated as critical business signals and incorporated into the "Risk Prediction" and "Fraud Analysis" sections for deep investigation. These points are crucial clues for analyzing loss causes, identifying key customers, or potential fraud.
    *   **Ensuring Model Robustness**: For predictive modeling, ensemble tree models like `RandomForest` and `Gradient Boosting`, which are robust to outliers, were selected to ensure overall model stability.

**Conclusion**: Outlier handling was guided by business insight, transforming potential "noise" into valuable objects of analysis, highlighting the diagnostic value of data analytics.

### 3.4 Data Patterns
Exploratory analysis revealed the following core patterns within the data, providing direct insights for business decision-making:

*   **Temporal Patterns**:
    *   **Growth & Seasonality**: Sales show a clear year-over-year growth trend with pronounced seasonal fluctuations (confirmed via Prophet model decomposition).
    *   **Diurnal Pattern**: Order volume follows a regular pattern throughout the 24-hour day, with higher activity during daytime hours.
*   **Product & Sales Patterns**:
    *   **Concentration at the Head**: A small number of products account for the vast majority of sales, with a clear best-seller list.
    *   **Category Focus**: Sporting goods and fitness equipment are core sales and profit drivers, but also categories with higher loss risks.
*   **Geographic & Risk Patterns**:
    *   **Clear Market Hierarchy**: The US market is the absolute core, followed by markets like Latin America and Europe.
    *   **Pronounced Risk Geography**: Delay rates, order loss rates, and fraud proportions vary significantly across regions (e.g., Central Africa and Eastern Europe score high on multiple risk indicators).
    *   **Shipping Mode Impact**: The delay risk for `Standard Class` shipping is significantly higher than for `First Class`.
*   **Customer Behavior Patterns**:
    *   **Clear Value Segmentation**: Through RFM (Recency, Frequency, Monetary) analysis, customers were effectively segmented into groups such as "Important Value Customers," "General Development Customers," and "Customers to Retain," showing a pyramid-shaped value distribution.

The preprocessed dataset is of high quality. Missing values have been effectively handled, class distributions reflect business reality, outliers have been transformed into analytical assets, and the data contains rich, clear business patterns. This dataset is now ready for direct use in advanced statistical analysis, visualization, and machine learning modeling.

---

## 4. Ethical and Legal Considerations

Although the dataset is publicly available, it still contains customer-related information. Therefore, ethical and legal considerations are important during the research process.

### 4.1 Privacy Protection

The dataset contains potentially sensitive fields such as:

- Customer Email
- Customer Street
- Customer Password
- Customer Name

To protect user privacy, this project will:

- Avoid publishing personally identifiable information (PII)
- Remove or anonymize sensitive attributes when necessary
- Avoid attempting user identity reconstruction

### 4.2 Legal and Responsible Usage

This dataset will only be used for:

- Academic research
- Course projects
- Educational data analysis purposes

The dataset will not be used for unauthorized commercial activities.

In addition, the project will comply with the dataset usage policies and citation requirements provided by the Mendeley Data platform.

### 4.3 Fairness and Bias Issues

When building models for:

- Fraud detection
- Customer value analysis
- Customer classification

there may be risks of algorithmic bias toward specific regions, countries, or customer groups.

Therefore, the project will:

- Analyze feature bias carefully
- Reduce unfair prediction risks
- Improve model interpretability whenever possible

### 4.4 Data Security and Reproducibility

All preprocessing, modeling, and analytical procedures will be implemented and recorded using Python and Jupyter Notebook.

This helps ensure:

- Research transparency
- Experimental reproducibility
- Academic reliability
- Clear documentation of analytical procedures
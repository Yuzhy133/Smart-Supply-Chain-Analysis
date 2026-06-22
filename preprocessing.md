## Preprocessing Log


### Catalogue

1. [Dataset overview](#1-dataset-overview)

2. [Dataset descriptive statistics](#2-dataset-descriptive-statistics)

3. [Column name standardization](#3-column-name-standardization)

4. [Data type conversion](#4-data-type-conversion)

5. [Data filtering](#5-data-filtering)

6. [Missing value handling](#6-missing-value-handling)

7. [Feature engineering](#7-feature-engineering)


```python
import pandas as pd
import numpy as np
```

<a id="1-dataset-overview"></a>
## 1.Dataset overview
Use latin1 encoding instead of the default utf-8, as the original data contains special characters (such as the ampersand &)
Ensure that special symbols can be parsed correctly


```python
data = pd.read_csv(r'DataCoSupplyChainDataset.csv', encoding='latin1')
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Type</th>
      <th>Days for shipping (real)</th>
      <th>Days for shipment (scheduled)</th>
      <th>Benefit per order</th>
      <th>Sales per customer</th>
      <th>Delivery Status</th>
      <th>Late_delivery_risk</th>
      <th>Category Id</th>
      <th>Category Name</th>
      <th>Customer City</th>
      <th>...</th>
      <th>Order Zipcode</th>
      <th>Product Card Id</th>
      <th>Product Category Id</th>
      <th>Product Description</th>
      <th>Product Image</th>
      <th>Product Name</th>
      <th>Product Price</th>
      <th>Product Status</th>
      <th>shipping date (DateOrders)</th>
      <th>Shipping Mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>4</td>
      <td>91.250000</td>
      <td>314.640015</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.750000</td>
      <td>0</td>
      <td>2/3/2018 22:56</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRANSFER</td>
      <td>5</td>
      <td>4</td>
      <td>-249.089996</td>
      <td>311.359985</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.750000</td>
      <td>0</td>
      <td>1/18/2018 12:27</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CASH</td>
      <td>4</td>
      <td>4</td>
      <td>-247.779999</td>
      <td>309.720001</td>
      <td>Shipping on time</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>San Jose</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.750000</td>
      <td>0</td>
      <td>1/17/2018 12:06</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>4</td>
      <td>22.860001</td>
      <td>304.809998</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.750000</td>
      <td>0</td>
      <td>1/16/2018 11:45</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAYMENT</td>
      <td>2</td>
      <td>4</td>
      <td>134.210007</td>
      <td>298.250000</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.750000</td>
      <td>0</td>
      <td>1/15/2018 11:24</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>180514</th>
      <td>CASH</td>
      <td>4</td>
      <td>4</td>
      <td>40.000000</td>
      <td>399.980011</td>
      <td>Shipping on time</td>
      <td>0</td>
      <td>45</td>
      <td>Fishing</td>
      <td>Brooklyn</td>
      <td>...</td>
      <td>NaN</td>
      <td>1004</td>
      <td>45</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Field+%26+Stre...</td>
      <td>Field &amp; Stream Sportsman 16 Gun Fire Safe</td>
      <td>399.980011</td>
      <td>0</td>
      <td>1/20/2016 3:40</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>180515</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>2</td>
      <td>-613.770019</td>
      <td>395.980011</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>45</td>
      <td>Fishing</td>
      <td>Bakersfield</td>
      <td>...</td>
      <td>NaN</td>
      <td>1004</td>
      <td>45</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Field+%26+Stre...</td>
      <td>Field &amp; Stream Sportsman 16 Gun Fire Safe</td>
      <td>399.980011</td>
      <td>0</td>
      <td>1/19/2016 1:34</td>
      <td>Second Class</td>
    </tr>
    <tr>
      <th>180516</th>
      <td>TRANSFER</td>
      <td>5</td>
      <td>4</td>
      <td>141.110001</td>
      <td>391.980011</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>45</td>
      <td>Fishing</td>
      <td>Bristol</td>
      <td>...</td>
      <td>NaN</td>
      <td>1004</td>
      <td>45</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Field+%26+Stre...</td>
      <td>Field &amp; Stream Sportsman 16 Gun Fire Safe</td>
      <td>399.980011</td>
      <td>0</td>
      <td>1/20/2016 21:00</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>180517</th>
      <td>PAYMENT</td>
      <td>3</td>
      <td>4</td>
      <td>186.229996</td>
      <td>387.980011</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>45</td>
      <td>Fishing</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1004</td>
      <td>45</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Field+%26+Stre...</td>
      <td>Field &amp; Stream Sportsman 16 Gun Fire Safe</td>
      <td>399.980011</td>
      <td>0</td>
      <td>1/18/2016 20:18</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>180518</th>
      <td>PAYMENT</td>
      <td>4</td>
      <td>4</td>
      <td>168.949997</td>
      <td>383.980011</td>
      <td>Shipping on time</td>
      <td>0</td>
      <td>45</td>
      <td>Fishing</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1004</td>
      <td>45</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Field+%26+Stre...</td>
      <td>Field &amp; Stream Sportsman 16 Gun Fire Safe</td>
      <td>399.980011</td>
      <td>0</td>
      <td>1/19/2016 18:54</td>
      <td>Standard Class</td>
    </tr>
  </tbody>
</table>
<p>180519 rows × 53 columns</p>
</div>



<a id="2-dataset-descriptive-statistics"></a>
## 2. Dataset descriptive statistics
Generate statistical summaries of numerical features using `data.describe()`


```python
describe = data.describe()
describe
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Days for shipping (real)</th>
      <th>Days for shipment (scheduled)</th>
      <th>Benefit per order</th>
      <th>Sales per customer</th>
      <th>Late_delivery_risk</th>
      <th>Category Id</th>
      <th>Customer Id</th>
      <th>Customer Zipcode</th>
      <th>Department Id</th>
      <th>Latitude</th>
      <th>...</th>
      <th>Order Item Quantity</th>
      <th>Sales</th>
      <th>Order Item Total</th>
      <th>Order Profit Per Order</th>
      <th>Order Zipcode</th>
      <th>Product Card Id</th>
      <th>Product Category Id</th>
      <th>Product Description</th>
      <th>Product Price</th>
      <th>Product Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>180516.000000</td>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>...</td>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>24840.000000</td>
      <td>180519.000000</td>
      <td>180519.000000</td>
      <td>0.0</td>
      <td>180519.000000</td>
      <td>180519.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.497654</td>
      <td>2.931847</td>
      <td>21.974989</td>
      <td>183.107609</td>
      <td>0.548291</td>
      <td>31.851451</td>
      <td>6691.379495</td>
      <td>35921.126914</td>
      <td>5.443460</td>
      <td>29.719955</td>
      <td>...</td>
      <td>2.127638</td>
      <td>203.772096</td>
      <td>183.107609</td>
      <td>21.974989</td>
      <td>55426.132327</td>
      <td>692.509764</td>
      <td>31.851451</td>
      <td>NaN</td>
      <td>141.232550</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.623722</td>
      <td>1.374449</td>
      <td>104.433526</td>
      <td>120.043670</td>
      <td>0.497664</td>
      <td>15.640064</td>
      <td>4162.918106</td>
      <td>37542.461122</td>
      <td>1.629246</td>
      <td>9.813646</td>
      <td>...</td>
      <td>1.453451</td>
      <td>132.273077</td>
      <td>120.043670</td>
      <td>104.433526</td>
      <td>31919.279101</td>
      <td>336.446807</td>
      <td>15.640064</td>
      <td>NaN</td>
      <td>139.732492</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-4274.979980</td>
      <td>7.490000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>603.000000</td>
      <td>2.000000</td>
      <td>-33.937553</td>
      <td>...</td>
      <td>1.000000</td>
      <td>9.990000</td>
      <td>7.490000</td>
      <td>-4274.979980</td>
      <td>1040.000000</td>
      <td>19.000000</td>
      <td>2.000000</td>
      <td>NaN</td>
      <td>9.990000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>7.000000</td>
      <td>104.379997</td>
      <td>0.000000</td>
      <td>18.000000</td>
      <td>3258.500000</td>
      <td>725.000000</td>
      <td>4.000000</td>
      <td>18.265432</td>
      <td>...</td>
      <td>1.000000</td>
      <td>119.980003</td>
      <td>104.379997</td>
      <td>7.000000</td>
      <td>23464.000000</td>
      <td>403.000000</td>
      <td>18.000000</td>
      <td>NaN</td>
      <td>50.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>31.520000</td>
      <td>163.990005</td>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>6457.000000</td>
      <td>19380.000000</td>
      <td>5.000000</td>
      <td>33.144863</td>
      <td>...</td>
      <td>1.000000</td>
      <td>199.919998</td>
      <td>163.990005</td>
      <td>31.520000</td>
      <td>59405.000000</td>
      <td>627.000000</td>
      <td>29.000000</td>
      <td>NaN</td>
      <td>59.990002</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>5.000000</td>
      <td>4.000000</td>
      <td>64.800003</td>
      <td>247.399994</td>
      <td>1.000000</td>
      <td>45.000000</td>
      <td>9779.000000</td>
      <td>78207.000000</td>
      <td>7.000000</td>
      <td>39.279617</td>
      <td>...</td>
      <td>3.000000</td>
      <td>299.950012</td>
      <td>247.399994</td>
      <td>64.800003</td>
      <td>90008.000000</td>
      <td>1004.000000</td>
      <td>45.000000</td>
      <td>NaN</td>
      <td>199.990005</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>911.799988</td>
      <td>1939.989990</td>
      <td>1.000000</td>
      <td>76.000000</td>
      <td>20757.000000</td>
      <td>99205.000000</td>
      <td>12.000000</td>
      <td>48.781933</td>
      <td>...</td>
      <td>5.000000</td>
      <td>1999.989990</td>
      <td>1939.989990</td>
      <td>911.799988</td>
      <td>99301.000000</td>
      <td>1363.000000</td>
      <td>76.000000</td>
      <td>NaN</td>
      <td>1999.989990</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 29 columns</p>
</div>



<a id="3-column-name-standardization"></a>
## 3. Column name standardization

Original column name issues: 
- Including special characters such as spaces and parentheses 
- Inconsistent case 
- Not conducive to subsequent programming references

Preprocessing:
- Unified naming convention (lowercase + underscore), compliant with Python PEP8 coding standards
- Remove special characters to avoid code errors
- Improve code readability and maintainability



```python
# Modify column header format
data.columns = data.columns.str.lower().str.strip().str.replace(' ', '_').str.replace(r'[()]', '', regex=True)
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>days_for_shipping_real</th>
      <th>days_for_shipment_scheduled</th>
      <th>benefit_per_order</th>
      <th>sales_per_customer</th>
      <th>delivery_status</th>
      <th>late_delivery_risk</th>
      <th>category_id</th>
      <th>category_name</th>
      <th>customer_city</th>
      <th>...</th>
      <th>order_zipcode</th>
      <th>product_card_id</th>
      <th>product_category_id</th>
      <th>product_description</th>
      <th>product_image</th>
      <th>product_name</th>
      <th>product_price</th>
      <th>product_status</th>
      <th>shipping_date_dateorders</th>
      <th>shipping_mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>4</td>
      <td>91.250000</td>
      <td>314.640015</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.750000</td>
      <td>0</td>
      <td>2/3/2018 22:56</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRANSFER</td>
      <td>5</td>
      <td>4</td>
      <td>-249.089996</td>
      <td>311.359985</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.750000</td>
      <td>0</td>
      <td>1/18/2018 12:27</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CASH</td>
      <td>4</td>
      <td>4</td>
      <td>-247.779999</td>
      <td>309.720001</td>
      <td>Shipping on time</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>San Jose</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.750000</td>
      <td>0</td>
      <td>1/17/2018 12:06</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>4</td>
      <td>22.860001</td>
      <td>304.809998</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Los Angeles</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.750000</td>
      <td>0</td>
      <td>1/16/2018 11:45</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PAYMENT</td>
      <td>2</td>
      <td>4</td>
      <td>134.210007</td>
      <td>298.250000</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>73</td>
      <td>Sporting Goods</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1360</td>
      <td>73</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Smart+watch</td>
      <td>Smart watch</td>
      <td>327.750000</td>
      <td>0</td>
      <td>1/15/2018 11:24</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>180514</th>
      <td>CASH</td>
      <td>4</td>
      <td>4</td>
      <td>40.000000</td>
      <td>399.980011</td>
      <td>Shipping on time</td>
      <td>0</td>
      <td>45</td>
      <td>Fishing</td>
      <td>Brooklyn</td>
      <td>...</td>
      <td>NaN</td>
      <td>1004</td>
      <td>45</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Field+%26+Stre...</td>
      <td>Field &amp; Stream Sportsman 16 Gun Fire Safe</td>
      <td>399.980011</td>
      <td>0</td>
      <td>1/20/2016 3:40</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>180515</th>
      <td>DEBIT</td>
      <td>3</td>
      <td>2</td>
      <td>-613.770019</td>
      <td>395.980011</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>45</td>
      <td>Fishing</td>
      <td>Bakersfield</td>
      <td>...</td>
      <td>NaN</td>
      <td>1004</td>
      <td>45</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Field+%26+Stre...</td>
      <td>Field &amp; Stream Sportsman 16 Gun Fire Safe</td>
      <td>399.980011</td>
      <td>0</td>
      <td>1/19/2016 1:34</td>
      <td>Second Class</td>
    </tr>
    <tr>
      <th>180516</th>
      <td>TRANSFER</td>
      <td>5</td>
      <td>4</td>
      <td>141.110001</td>
      <td>391.980011</td>
      <td>Late delivery</td>
      <td>1</td>
      <td>45</td>
      <td>Fishing</td>
      <td>Bristol</td>
      <td>...</td>
      <td>NaN</td>
      <td>1004</td>
      <td>45</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Field+%26+Stre...</td>
      <td>Field &amp; Stream Sportsman 16 Gun Fire Safe</td>
      <td>399.980011</td>
      <td>0</td>
      <td>1/20/2016 21:00</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>180517</th>
      <td>PAYMENT</td>
      <td>3</td>
      <td>4</td>
      <td>186.229996</td>
      <td>387.980011</td>
      <td>Advance shipping</td>
      <td>0</td>
      <td>45</td>
      <td>Fishing</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1004</td>
      <td>45</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Field+%26+Stre...</td>
      <td>Field &amp; Stream Sportsman 16 Gun Fire Safe</td>
      <td>399.980011</td>
      <td>0</td>
      <td>1/18/2016 20:18</td>
      <td>Standard Class</td>
    </tr>
    <tr>
      <th>180518</th>
      <td>PAYMENT</td>
      <td>4</td>
      <td>4</td>
      <td>168.949997</td>
      <td>383.980011</td>
      <td>Shipping on time</td>
      <td>0</td>
      <td>45</td>
      <td>Fishing</td>
      <td>Caguas</td>
      <td>...</td>
      <td>NaN</td>
      <td>1004</td>
      <td>45</td>
      <td>NaN</td>
      <td>http://images.acmesports.sports/Field+%26+Stre...</td>
      <td>Field &amp; Stream Sportsman 16 Gun Fire Safe</td>
      <td>399.980011</td>
      <td>0</td>
      <td>1/19/2016 18:54</td>
      <td>Standard Class</td>
    </tr>
  </tbody>
</table>
<p>180519 rows × 53 columns</p>
</div>



<a id="4-data-type-conversion"></a>
## 4. Data type conversion
Preprocessing:
- Facilitate time series analysis and date feature extraction
- Support date range filtering operation
- Calculable time difference (such as transportation duration)


```python
# Change the date to datetime type
data['order_date_dateorders'] = pd.to_datetime(data['order_date_dateorders'])
data['shipping_date_dateorders'] = pd.to_datetime(data['shipping_date_dateorders'])
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 180519 entries, 0 to 180518
    Data columns (total 53 columns):
     #   Column                       Non-Null Count   Dtype         
    ---  ------                       --------------   -----         
     0   type                         180519 non-null  object        
     1   days_for_shipping_real       180519 non-null  int64         
     2   days_for_shipment_scheduled  180519 non-null  int64         
     3   benefit_per_order            180519 non-null  float64       
     4   sales_per_customer           180519 non-null  float64       
     5   delivery_status              180519 non-null  object        
     6   late_delivery_risk           180519 non-null  int64         
     7   category_id                  180519 non-null  int64         
     8   category_name                180519 non-null  object        
     9   customer_city                180519 non-null  object        
     10  customer_country             180519 non-null  object        
     11  customer_email               180519 non-null  object        
     12  customer_fname               180519 non-null  object        
     13  customer_id                  180519 non-null  int64         
     14  customer_lname               180511 non-null  object        
     15  customer_password            180519 non-null  object        
     16  customer_segment             180519 non-null  object        
     17  customer_state               180519 non-null  object        
     18  customer_street              180519 non-null  object        
     19  customer_zipcode             180516 non-null  float64       
     20  department_id                180519 non-null  int64         
     21  department_name              180519 non-null  object        
     22  latitude                     180519 non-null  float64       
     23  longitude                    180519 non-null  float64       
     24  market                       180519 non-null  object        
     25  order_city                   180519 non-null  object        
     26  order_country                180519 non-null  object        
     27  order_customer_id            180519 non-null  int64         
     28  order_date_dateorders        180519 non-null  datetime64[ns]
     29  order_id                     180519 non-null  int64         
     30  order_item_cardprod_id       180519 non-null  int64         
     31  order_item_discount          180519 non-null  float64       
     32  order_item_discount_rate     180519 non-null  float64       
     33  order_item_id                180519 non-null  int64         
     34  order_item_product_price     180519 non-null  float64       
     35  order_item_profit_ratio      180519 non-null  float64       
     36  order_item_quantity          180519 non-null  int64         
     37  sales                        180519 non-null  float64       
     38  order_item_total             180519 non-null  float64       
     39  order_profit_per_order       180519 non-null  float64       
     40  order_region                 180519 non-null  object        
     41  order_state                  180519 non-null  object        
     42  order_status                 180519 non-null  object        
     43  order_zipcode                24840 non-null   float64       
     44  product_card_id              180519 non-null  int64         
     45  product_category_id          180519 non-null  int64         
     46  product_description          0 non-null       float64       
     47  product_image                180519 non-null  object        
     48  product_name                 180519 non-null  object        
     49  product_price                180519 non-null  float64       
     50  product_status               180519 non-null  int64         
     51  shipping_date_dateorders     180519 non-null  datetime64[ns]
     52  shipping_mode                180519 non-null  object        
    dtypes: datetime64[ns](2), float64(15), int64(14), object(22)
    memory usage: 73.0+ MB
    

<a id="5-data-filtering"></a>
## 5. Data filtering
Preprocessing operation:
- Remove data after October 1, 2017 (possibly incomplete data collected in the later stage)
- Ensure the stability of time series analysis
- Avoid contamination of the training model by future data
- Use `.copy()` to avoid the SettingWithCopyWarning


```python
cutoff_date = pd.Timestamp('2017-10-01')
data = data[data['order_date_dateorders'] < cutoff_date].copy()
```


```python
data.isnull().sum()
```




    type                                0
    days_for_shipping_real              0
    days_for_shipment_scheduled         0
    benefit_per_order                   0
    sales_per_customer                  0
    delivery_status                     0
    late_delivery_risk                  0
    category_id                         0
    category_name                       0
    customer_city                       0
    customer_country                    0
    customer_email                      0
    customer_fname                      0
    customer_id                         0
    customer_lname                      0
    customer_password                   0
    customer_segment                    0
    customer_state                      0
    customer_street                     0
    customer_zipcode                    0
    department_id                       0
    department_name                     0
    latitude                            0
    longitude                           0
    market                              0
    order_city                          0
    order_country                       0
    order_customer_id                   0
    order_date_dateorders               0
    order_id                            0
    order_item_cardprod_id              0
    order_item_discount                 0
    order_item_discount_rate            0
    order_item_id                       0
    order_item_product_price            0
    order_item_profit_ratio             0
    order_item_quantity                 0
    sales                               0
    order_item_total                    0
    order_profit_per_order              0
    order_region                        0
    order_state                         0
    order_status                        0
    order_zipcode                  147122
    product_card_id                     0
    product_category_id                 0
    product_description            171962
    product_image                       0
    product_name                        0
    product_price                       0
    product_status                      0
    shipping_date_dateorders            0
    shipping_mode                       0
    dtype: int64



<a id="6-missing-value-handling"></a>
## 6. Missing value handling
Preprocessing:
- Count the missing values based on data.isnull().sum()
- product_description: completely empty (171,962/171,962), lacking any informational value
- order_zipcode: 81.5% missing rate, and there are already location information such as order_city, order_state, order_region, etc
- customer_zipcode: 99.97% missing, with extremely low information content
- customer_email & customer_password: personal privacy information, should not be included in analysis



```python
data = data.drop(columns = ['product_description', 'order_zipcode', 'customer_zipcode', 
                            'customer_email', 'customer_password'])
```


```python
data.isnull().sum()
```




    type                           0
    days_for_shipping_real         0
    days_for_shipment_scheduled    0
    benefit_per_order              0
    sales_per_customer             0
    delivery_status                0
    late_delivery_risk             0
    category_id                    0
    category_name                  0
    customer_city                  0
    customer_country               0
    customer_fname                 0
    customer_id                    0
    customer_lname                 0
    customer_segment               0
    customer_state                 0
    customer_street                0
    department_id                  0
    department_name                0
    latitude                       0
    longitude                      0
    market                         0
    order_city                     0
    order_country                  0
    order_customer_id              0
    order_date_dateorders          0
    order_id                       0
    order_item_cardprod_id         0
    order_item_discount            0
    order_item_discount_rate       0
    order_item_id                  0
    order_item_product_price       0
    order_item_profit_ratio        0
    order_item_quantity            0
    sales                          0
    order_item_total               0
    order_profit_per_order         0
    order_region                   0
    order_state                    0
    order_status                   0
    product_card_id                0
    product_category_id            0
    product_image                  0
    product_name                   0
    product_price                  0
    product_status                 0
    shipping_date_dateorders       0
    shipping_mode                  0
    dtype: int64



<a id="7-feature-engineering"></a>
## 7. Feature engineering 
- Extracts multiple time-dimensional features from order_date_dateorders. 
- Time features are crucial for prediction tasks such as sales forecasting and demand forecasting. 
- Decomposing the time dimension facilitates the capture of periodic patterns and supports multi-granularity time series analysis (day, month, year, week)
- ORDER_WEEKDAY can analyze weekend effects


```python
data['ORDER_YEAR'] = data['order_date_dateorders'].dt.year
data['ORDER_MONTH'] = data['order_date_dateorders'].dt.month
data['ORDER_DAY'] = data['order_date_dateorders'].dt.day
data['ORDER_WEEKDAY'] = data['order_date_dateorders'].dt.weekday
data['ORDER_DATE'] = data['order_date_dateorders'].dt.date
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Index: 171962 entries, 48 to 180518
    Data columns (total 53 columns):
     #   Column                       Non-Null Count   Dtype         
    ---  ------                       --------------   -----         
     0   type                         171962 non-null  object        
     1   days_for_shipping_real       171962 non-null  int64         
     2   days_for_shipment_scheduled  171962 non-null  int64         
     3   benefit_per_order            171962 non-null  float64       
     4   sales_per_customer           171962 non-null  float64       
     5   delivery_status              171962 non-null  object        
     6   late_delivery_risk           171962 non-null  int64         
     7   category_id                  171962 non-null  int64         
     8   category_name                171962 non-null  object        
     9   customer_city                171962 non-null  object        
     10  customer_country             171962 non-null  object        
     11  customer_fname               171962 non-null  object        
     12  customer_id                  171962 non-null  int64         
     13  customer_lname               171962 non-null  object        
     14  customer_segment             171962 non-null  object        
     15  customer_state               171962 non-null  object        
     16  customer_street              171962 non-null  object        
     17  department_id                171962 non-null  int64         
     18  department_name              171962 non-null  object        
     19  latitude                     171962 non-null  float64       
     20  longitude                    171962 non-null  float64       
     21  market                       171962 non-null  object        
     22  order_city                   171962 non-null  object        
     23  order_country                171962 non-null  object        
     24  order_customer_id            171962 non-null  int64         
     25  order_date_dateorders        171962 non-null  datetime64[ns]
     26  order_id                     171962 non-null  int64         
     27  order_item_cardprod_id       171962 non-null  int64         
     28  order_item_discount          171962 non-null  float64       
     29  order_item_discount_rate     171962 non-null  float64       
     30  order_item_id                171962 non-null  int64         
     31  order_item_product_price     171962 non-null  float64       
     32  order_item_profit_ratio      171962 non-null  float64       
     33  order_item_quantity          171962 non-null  int64         
     34  sales                        171962 non-null  float64       
     35  order_item_total             171962 non-null  float64       
     36  order_profit_per_order       171962 non-null  float64       
     37  order_region                 171962 non-null  object        
     38  order_state                  171962 non-null  object        
     39  order_status                 171962 non-null  object        
     40  product_card_id              171962 non-null  int64         
     41  product_category_id          171962 non-null  int64         
     42  product_image                171962 non-null  object        
     43  product_name                 171962 non-null  object        
     44  product_price                171962 non-null  float64       
     45  product_status               171962 non-null  int64         
     46  shipping_date_dateorders     171962 non-null  datetime64[ns]
     47  shipping_mode                171962 non-null  object        
     48  ORDER_YEAR                   171962 non-null  int32         
     49  ORDER_MONTH                  171962 non-null  int32         
     50  ORDER_DAY                    171962 non-null  int32         
     51  ORDER_WEEKDAY                171962 non-null  int32         
     52  ORDER_DATE                   171962 non-null  object        
    dtypes: datetime64[ns](2), float64(12), int32(4), int64(14), object(21)
    memory usage: 68.2+ MB
    


```python
data.to_csv('cleaned_data.csv')
```

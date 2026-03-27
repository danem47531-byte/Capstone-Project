# 🛒 Flipkart Men's Topwear — End-to-End Capstone Project

![Python](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Tool-Jupyter%20Notebook-F37626?logo=jupyter&logoColor=white) ![Power BI](https://img.shields.io/badge/Dashboard-Power%20BI-F2C811?logo=powerbi&logoColor=black) ![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## 📌 Project Overview

This is a **full end-to-end capstone project** on Flipkart's Men's Topwear category. It covers the complete data analytics and machine learning pipeline — from **live web scraping** of real product data directly from Flipkart, through **data cleaning**, **exploratory data analysis (EDA)**, a **content-based recommendation system**, a **price prediction ML model**, and finally an **interactive Power BI dashboard**.

The project uses **1,000 real scraped products** from Flipkart's Men's Topwear section to uncover pricing strategies, discount patterns, brand behavior, and customer savings.

---

## 🗂️ Dataset

The dataset was **self-scraped** from Flipkart using `BeautifulSoup` and `requests`. It contains **1,000 men's topwear products** with the following fields:

| Column | Description |
|---|---|
| `name` | Product name |
| `company` | Brand/seller name |
| `price` | Discounted selling price (₹) |
| `original_price` | MRP / original listed price (₹) |
| `discount` | Discount percentage as scraped (e.g., "70% off") |
| `offers` | Active offers (e.g., "Hot Deal", "Buy 2 save 5%") |
| `url` | Direct product URL on Flipkart |
| `page` | Flipkart page number the product was found on |

**Engineered Features (added during cleaning):**

| Feature | Description |
|---|---|
| `discount_pct` | Calculated discount % = `(original_price - price) / original_price × 100` |
| `savings` | Absolute savings = `original_price - price` |
| `price_band` | Budget (< ₹500) / Mid-Range (₹500–₹1500) / Premium (> ₹1500) |
| `high_discount` | Binary flag — 1 if discount ≥ 50%, else 0 |

---

## 🛠️ Libraries & Tools Used

```python
# Data Collection
import requests
from bs4 import BeautifulSoup

# Data Processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
```

**Dashboard:** Microsoft Power BI

---

## 🔍 Phase 1 — Web Scraping

- Built a **custom Flipkart scraper** using `BeautifulSoup` and `requests`
- Scraped the **Men's Topwear category** across multiple pages
- Added **random delays** (`time.sleep(random.uniform(2, 4))`) to avoid bot detection
- Used rotating page parameters via `urllib.parse` to paginate through results
- Successfully extracted **1,000 product records** and saved to `flipkart_mens_topwear_1000_records.csv`
- Scraped fields: Product name, brand/company, selling price, original price, discount %, active offers, product URL, page number

---

## 🧹 Phase 2 — Data Cleaning & Preprocessing

- Stripped **currency symbols and special characters** from `price` and `original_price` columns using regex, converted to `float`
- Extracted **numeric discount values** from strings like "70% off" using `.str.extract(r'(\d+)')`
- Handled **missing values**:
  - `original_price` → filled with **median**
  - `discount` → filled with **0**
  - `offers` → filled with **"No Offer"**
- Dropped the `size` column (all values were "N/A")
- **Engineered 4 new features**: `discount_pct`, `savings`, `price_band`, `high_discount`

---

## 📊 Phase 3 — Exploratory Data Analysis (EDA)

### Summary Statistics
- **Average selling price:** ₹347 | **Median:** ₹310
- **Mean original price:** ₹1,560 — massive gap highlighting discount-driven pricing
- **Average discount:** over **75%**
- **Average customer savings:** ~₹1,213 per product

### Trend Analysis
- Grouped average price and discount by **company** and **page number**
- Identified **top brands** by average price and average savings
- Analyzed **price band distribution** (Budget / Mid-Range / Premium)
- Examined **high-discount product frequency**

### Visualizations

| Chart | Type | Insight |
|---|---|---|
| Average Price by Top Companies | Bar Chart | Brand-wise pricing comparison |
| Average Price Trend by Page | Line Chart | Price variation across listing pages |
| Price vs Discount Percentage | Scatter Plot | Relationship between price and discount depth |
| Boxplot of Product Prices | Box Plot | Price spread and outliers |
| Boxplot of Original Prices | Box Plot | MRP spread and outlier detection |
| Boxplot of Discount Percentage | Box Plot | Discount distribution and extremes |
| Price vs Original Price vs Discount % | Combined Box Plot | Side-by-side comparison |
| Correlation Heatmap | Heatmap | Correlation between price, discount, savings, page |

### Key EDA Finding
> *"The Flipkart men's topwear marketplace is strongly driven by a value-for-money and discount-focused strategy. With an average discount of over 75% and average savings of ₹1,213, perceived savings play a major role in influencing purchase decisions. Customer behavior is driven more by visible discounts than brand prestige — making promotional pricing the central competitive strategy."*

---

## 🤖 Phase 4 — Machine Learning Models

### Model 1: Content-Based Recommendation System

Built a **product recommendation engine** using TF-IDF vectorization and cosine similarity:

```python
# Combine product features into a single text field
df_model['combined_features'] = (
    df_model['name'] + " " +
    df_model['company'] + " " +
    df_model['price_band'] + " " +
    df_model['discount_pct']
)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
feature_matrix = vectorizer.fit_transform(df_model['combined_features'])

# Cosine Similarity Matrix
similarity_matrix = cosine_similarity(feature_matrix)
```

- A `recommend(query, top_n=5)` function accepts a **product name, brand, price, or discount** as input
- Returns the **top 5 most similar products** based on cosine similarity scores
- Supports flexible search across name, company, price, and discount fields

### Model 2: Price Prediction — Linear Regression

Built a **price prediction model** using Linear Regression:

**Features used:** `original_price`, `discount_pct`, `company` (one-hot encoded), `page`

**Target variable:** `price` (discounted selling price)

```python
X = pd.get_dummies(X, columns=['company'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
```

### 📈 Model Performance

| Metric | Value |
|---|---|
| **R² Score (Test)** | **0.9339** |
| **RMSE** | **29.97** |
| **Training Score** | 0.9632 |
| **Testing Score** | 0.9340 |

> *The high R² score (0.93) confirms strong linear dependency between selling price and features like original price and discount percentage — exactly what's expected in a pricing model.*

**Visualizations:**
- Actual vs Predicted Price scatter plot
- Prediction Error Distribution histogram
- Top feature coefficients table

---

## 📊 Phase 5 — Interactive Power BI Dashboard

An interactive **Power BI dashboard** (`Flipkart analysis dashboard.pbix`) was built to visualize all key insights:

- Price vs Discount trends across brands and pages
- Company-wise average savings and discount depth
- Price band distribution (Budget / Mid-Range / Premium)
- High-discount product concentration
- Interactive slicers for filtering by company, price band, and page

---

## 💡 Key Business Insights & Recommendations

**Key Insight:**
Customer purchase behavior in men's fashion on Flipkart is driven by **discount depth, mid-range pricing, and promotional offers** rather than brand premium positioning.

**Recommendations:**
- Flipkart should follow a **promotional-driven, value-focused strategy** where revenue is maximized through discount structures and mid-range pricing
- Focus on **multi-offer bundling** (e.g., "Buy 2 save 5%") to increase cart value
- High-discount items are effective marketing tools — use them to drive traffic and conversions
- Premium brands should be promoted separately using brand-awareness campaigns, not just discount-led listings

---

## 📁 Project Files

```
📦 Capstone Project
 ┣ 📁 All .ipynb files/
 ┃ ┣ 📓 Web scraping flipcart dataset.ipynb         # Phase 1 — Live web scraping
 ┃ ┣ 📓 Data Preprossing And Cleaning.ipynb         # Phase 2 — Data cleaning
 ┃ ┣ 📓 EDA Analysis.ipynb                          # Phase 3 — EDA & visualizations
 ┃ ┣ 📓 Machine Learning Model.ipynb                # Phase 4 — Recommendation + Price prediction
 ┃ ┗ 📓 Data Cleaning To Machine Learning.ipynb     # Full pipeline in one notebook
 ┣ 📁 CSV Files/
 ┃ ┣ 📄 flipkart_mens_topwear_1000_records.csv      # Raw scraped dataset (1000 products)
 ┃ ┣ 📄 flipcart clean data.csv                     # Cleaned dataset
 ┃ ┣ 📄 results.csv                                 # Actual vs Predicted prices
 ┃ ┣ 📄 coefficients.csv                            # Linear Regression feature coefficients
 ┃ ┣ 📄 company_savings.csv                         # Avg savings per brand
 ┃ ┣ 📄 company_discount_pact.csv                   # Avg discount per brand
 ┃ ┣ 📄 discount_pct.csv                            # Discount % per product
 ┃ ┣ 📄 high_discount.csv                           # High-discount product flags
 ┃ ┣ 📄 price_band.csv                              # Price band classification
 ┃ ┣ 📄 savings.csv                                 # Savings per product
 ┃ ┗ 📄 unique_price.csv                            # Unique price values
 ┣ 📊 Flipkart analysis dashboard.pbix              # Power BI interactive dashboard
 ┣ 📄 Capstone Project On Flipkart.pdf              # Full project report with screenshots
 ┗ 📄 Capstone Project On Flipkart.docx             # Word document version of report
---

## 🚀 How to Run

1. Clone this repository
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn requests beautifulsoup4 openpyxl
   ```
3. To use the pre-scraped data, load `CSV Files/flipkart_mens_topwear_1000_records.csv` directly
4. Run notebooks in this order:
   - `Data Preprossing And Cleaning.ipynb`
   - `EDA Analysis.ipynb`
   - `Machine Learning Model.ipynb`
   - OR run `Data Cleaning To Machine Learning.ipynb` for the complete pipeline in one file
5. Open `Flipkart analysis dashboard.pbix` in **Power BI Desktop** for the interactive dashboard

> **Note:** To re-run the web scraper, open `Web scraping flipcart dataset.ipynb`. Be aware that Flipkart may block automated requests — the scraper includes delays to reduce this risk.

---

## 👤 Author

### Tridev Pal
📍 Calcutta, West Bengal, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?logo=linkedin)](https://www.linkedin.com/in/tridev-pal-74575a379/)
> Feel free to connect or raise issues via GitHub if you have suggestions or improvements!

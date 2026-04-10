# ML PROJECT COLLECTION

A comprehensive collection of machine learning projects covering regression, classification, and analytics use cases. Each project includes notebook-based analysis and corresponding Streamlit apps for interactive plots and predictions.

## PROJECTS

1. Advertising Budget Prediction
- Notebook: KaushalSavaliya_Advertising_Budget.ipynb
- Streamlit app: streamlit_advertising_budget.py
- Target: Sales

2. Bike Sharing Analysis
- Notebook: KaushalSavaliya_Bike_Sharing_Analysis.ipynb
- Streamlit app: streamlit_bike_sharing.py
- Target: cnt or count

3. Credit Card Fraud Analysis
- Notebook: KaushalSavaliya_CreditCard.ipynb
- Streamlit app: streamlit_credit_card.py
- Target: Class

4. Diabetes Prediction
- Notebook: KaushalSavaliya_DiabitisPrediction.ipynb
- Streamlit app: streamlit_diabetes_prediction.py
- Target: Outcome

5. Food Delivery System Analysis
- Notebook: KaushalSavaliya_FoodDeliverySystem.ipynb
- Streamlit app: streamlit_food_delivery.py
- Typical targets: delivery time or segment labels

6. Health Insurance Analysis
- Notebook: KaushalSavaliya_Health_Insurance.ipynb
- Streamlit app: streamlit_health_insurance.py
- Target: charges

7. Online Shopping Analysis
- Notebook: KaushalSavaliya_OnlineShopping.ipynb
- Streamlit app: streamlit_online_shopping.py
- Target: Revenue

## STREAMLIT APPS

Shared core module:
- streamlit_project_core.py

Per-project app entry files:
- streamlit_advertising_budget.py
- streamlit_bike_sharing.py
- streamlit_credit_card.py
- streamlit_diabetes_prediction.py
- streamlit_food_delivery.py
- streamlit_health_insurance.py
- streamlit_online_shopping.py

## RUN

Install dependencies:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn
```

Run any app:

```bash
streamlit run streamlit_advertising_budget.py
```

Replace with any of the other app files as needed.

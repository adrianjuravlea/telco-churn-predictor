# Telco Customer Churn Prediction & Retention Strategy

**Business problem**: Telecom companies lose millions annually due to customer churn. Most retention campaigns are sent blindly, wasting marketing budget.

**Solution**: Predictive system that identifies customers likely to churn in the next 30 days and only targets high-risk ones.

**Financial impact**: For a company with 50,000 customers and 5% monthly churn, implementing this system could **retain ~$2.1M in annual revenue** (conservative estimate, full calculation in notebook).

**Key features**:
- Profit-optimized threshold (not accuracy!) → maximizes $ saved
- Cost-benefit analysis: $8 intervention cost vs $1,200 average CLV saved
- Best model: XGBoost/LightGBM with custom business metric
- Deployed as interactive Streamlit app

**Tech stack**: Python, Pandas, Scikit-learn, XGBoost, LightGBM, Streamlit

[▶️ Live Demo (Streamlit)](https://telco-churn-predictor-z9qnp8jjsxkpwt6jrm6vzi.streamlit.app/)  
[📊 Jupyter Notebook](notebook/Telco_Churn_Analysis.ipynb)  
[🚀 Streamlit App Code](app/app.py)
# Personalized Cosmetics‑Box Insights Dashboard

An end‑to‑end Streamlit application that explores, models, and segments the synthetic cosmetics‑survey dataset.

<p align="center">
  <img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="120" />
</p>

## 💡 Features
* **Multi‑tab UI** built with Streamlit’s native `st.tabs`.
* **Data‑visualisation hub**: 10+ descriptive plots & complex KPIs.
* **Classification lab**: K‑NN, Decision Tree, Random Forest, Gradient Boosting with metrics table, toggleable confusion‑matrix, and combined ROC.
* **Clustering studio**: interactive K‑Means with elbow plot, adjustable *k* slider, persona table, downloadable cluster‑tagged file.
* **Association rules**: Apriori on multi‑select columns, tunable support & confidence, top‑10 rules preview.
* **Regression corner**: Linear, Ridge, Lasso, Decision Tree Regressor with instant insights.
* **Data I/O**: upload new CSVs for prediction; one‑click download of results or cluster labels.
* Designed to be **Streamlit‑Cloud ready** — just push to GitHub and deploy.

## 🚀 Quick start

```bash
# 1. Clone repo & install deps
git clone <your‑repo‑url>
cd cosmetics_streamlit_dashboard
pip install -r requirements.txt

# 2. Run locally
streamlit run app.py
```

## 🌐 Streamlit Cloud

1. Push this folder to a public GitHub repo.  
2. Sign in to **share.streamlit.io** → “New app” → point to **app.py**.  
3. Add a **Secrets** section if you store the dataset privately; otherwise the default GitHub‑raw path will load it automatically.

## 📂 Data

* `data/cosmetics_survey_synthetic.csv` — sample dataset (1 200 rows).  
* You can replace it or provide a raw‑GitHub URL via the sidebar.

_Enjoy exploring personalized beauty consumer insights!_

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils import load_data, basic_preprocess

st.set_page_config(page_title="Cosmetics‑Box Insights", layout="wide")
st.title("🎁 Personalized Cosmetics‑Box Consumer Insights Dashboard")

# Sidebar data loader
st.sidebar.header("📂 Data source")
sample_path = "data/cosmetics_survey_synthetic.csv"
data_option = st.sidebar.radio(
    "Select data",
    ("Use sample data", "Load from GitHub raw URL", "Upload CSV"))

if data_option == "Use sample data":
    df = load_data(sample_path)
elif data_option == "Load from GitHub raw URL":
    url = st.sidebar.text_input("Paste raw GitHub URL", "")
    if url:
        df = load_data(url)
    else:
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Upload your CSV")
    if uploaded:
        df = pd.read_csv(uploaded)
    else:
        st.stop()

st.sidebar.success(f"Loaded {df.shape[0]:,} rows.")

tabs = st.tabs(["🔍 Data Visualisation", "🧮 Classification", "🎯 Clustering",
                "🤝 Association Rules", "📈 Regression"])

# =============== 1. DATA VISUALISATION =================
with tabs[0]:
    st.header("Descriptive Insights")
    sub_tabs = st.tabs([f"Insight {i}" for i in range(1,11)])
    # Insight 1: Age distribution
    with sub_tabs[0]:
        st.subheader("Age‑group distribution")
        fig = px.histogram(df, x="age_group", nbins=6, title="Respondent counts by age group")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Shows the concentration of survey participants across age cohorts.")
    # Insight 2: Monthly spend vs online freq
    with sub_tabs[1]:
        st.subheader("Monthly spend vs online buy frequency")
        fig = px.box(df, x="online_buy_freq", y="monthly_spend_usd", points="all",
                     title="Spend increases with purchasing frequency")
        st.plotly_chart(fig, use_container_width=True)
    # Additional insights 3‑10
    insights_map = {
        2: ("Income vs Subscription Likelihood", "personal_income_usd", "sub_likelihood"),
        3: ("Brand loyalty vs Referral likelihood", "brand_loyalty", "refer_friend"),
        4: ("Region vs Avg budget per box", "region", "budget_per_box"),
        5: ("Features importance heatmap", None, None),
        6: ("Spend distribution by Gender", "gender", "monthly_spend_usd"),
        7: ("Influence of social media creators", "social_influence", None),
        8: ("Payment model preference counts", "pay_freq_pref", None),
        9: ("Value of personalisation vs Likelihood to subscribe", "value_personalized","sub_likelihood"),
    }
    for idx, (title,x,y) in insights_map.items():
        with sub_tabs[idx]:
            st.subheader(title)
            if title=="Features importance heatmap":
                bin_df = df["seek_features"].str.get_dummies(sep=",")
                feat_counts = bin_df.sum().sort_values(ascending=False)
                fig = px.bar(feat_counts, title="Product feature demand (binary counts)")
                st.plotly_chart(fig, use_container_width=True)
            elif y:
                fig = px.scatter(df, x=x, y=y, color="age_group", trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.box(df, x=x, y=y) if y else px.histogram(df, x=x)
                st.plotly_chart(fig, use_container_width=True)
    # Insight 10
    with sub_tabs[9]:
        st.subheader("Top pain points")
        pain_counts = df["purchase_pain"].str.get_dummies(sep=",").sum().sort_values()
        fig = px.bar(pain_counts, orientation="h", title="Most cited shopping challenges")
        st.plotly_chart(fig, use_container_width=True)


# =============== 2. CLASSIFICATION =================
with tabs[1]:
    st.header("Predict “Likely to Subscribe” (≥4 on Likert)")
    target = "WillSubscribe"
    df[target] = (df["sub_likelihood"] >=4).astype(int)

    # Features and preprocessing
    X, y, preprocessor = basic_preprocess(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "K‑NN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    metrics_data = []
    for name, model in models.items():
        pipe = preprocessor | model
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        metrics_data.append((name, acc, prec, rec, f1))
        models[name] = pipe  # store fitted

    st.subheader("Performance Comparison")
    met_df = pd.DataFrame(metrics_data, columns=["Model","Accuracy","Precision","Recall","F1"])
    st.dataframe(met_df.style.format({c:"{:.2%}" for c in met_df.columns[1:]}), use_container_width=True)

    st.divider()
    sel_model = st.selectbox("Select model for confusion matrix & ROC", list(models.keys()))
    cm = confusion_matrix(y_test, models[sel_model].predict(X_test))
    st.subheader("Confusion Matrix")
    st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

    st.subheader("ROC curve")
    fig, ax = plt.subplots()
    for name, pipe in models.items():
        y_prob = pipe.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1],[0,1],"--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Upload new data for prediction")
    new_file = st.file_uploader("Upload CSV without target variable", key="pred_upload")
    if new_file:
        new_df = pd.read_csv(new_file)
        pred = models[sel_model].predict(new_df)
        new_df["Predicted_"+target] = pred
        st.write(new_df.head())
        st.download_button("Download predictions", new_df.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")

# =============== 3. CLUSTERING =================
with tabs[2]:
    st.header("K‑Means Customer Segmentation")
    k = st.slider("Number of clusters (k)", 2, 10, 4)
    num_cols = ["personal_income_usd","online_buy_freq","monthly_spend_usd",
                "budget_per_box","social_influence","brand_loyalty","value_personalized"]
    kmeans = KMeans(n_clusters=k, random_state=42)
    df_cluster = df.copy()
    df_cluster["cluster"] = kmeans.fit_predict(df_cluster[num_cols].fillna(0))
    st.write("Cluster counts", df_cluster["cluster"].value_counts().rename("count"))

    # Elbow plot pre-computed up to 10
    if "wss" not in st.session_state:
        inertias=[]
        for ki in range(1,11):
            km = KMeans(n_clusters=ki, random_state=42).fit(df[num_cols].fillna(0))
            inertias.append(km.inertia_)
        st.session_state["wss"] = inertias
    fig = px.line(x=range(1,11), y=st.session_state["wss"],
                  labels={"x":"k","y":"Within‑cluster Sum of Squares"},
                  title="Elbow method")
    st.plotly_chart(fig, use_container_width=True)

    # Persona table (mean of numeric cols)
    persona = df_cluster.groupby("cluster")[num_cols].mean().round(1)
    st.subheader("Cluster Personas (numeric means)")
    st.dataframe(persona)
    st.download_button("Download cluster‑tagged data",
                       df_cluster.to_csv(index=False).encode("utf-8"),
                       file_name="clustered_data.csv")

# =============== 4. ASSOCIATION RULES =================
with tabs[3]:
    st.header("Market‑Basket Insights via Apriori")
    cand_cols = ["fav_product_types","seek_features","purchase_pain"]
    sel_cols = st.multiselect("Select at least 2 columns", cand_cols, default=cand_cols[:2])
    if len(sel_cols) <2:
        st.stop()
    bin_df = pd.DataFrame()
    for col in sel_cols:
        bin_df = bin_df.join(df[col].str.get_dummies(sep=","), how="outer")
    min_sup = st.slider("Minimum support", 0.01, 0.5, 0.05, 0.01)
    min_conf = st.slider("Minimum confidence", 0.1, 1.0, 0.3, 0.05)
    freq_items = apriori(bin_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    top_rules = rules.sort_values("confidence", ascending=False).head(10)
    st.dataframe(top_rules[["antecedents","consequents","support","confidence","lift"]])

# =============== 5. REGRESSION =================
with tabs[4]:
    st.header("Budget per Box – Quick Regressors")
    target = "budget_per_box"
    num_cols = ["personal_income_usd","online_buy_freq","monthly_spend_usd",
                "social_influence","app_usage","value_personalized"]
    X = df[num_cols]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42)
    }
    reg_metrics=[]
    for name, model in reg_models.items():
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        reg_metrics.append((name, r2))
    reg_df = pd.DataFrame(reg_metrics, columns=["Model","R² on test"])
    st.dataframe(reg_df.style.format({"R² on test":"{:.2f}"}))

    best = reg_df.sort_values("R² on test", ascending=False).iloc[0,0]
    st.subheader(f"Best model: {best}")
    model = reg_models[best]
    # Plot actual vs predicted
    y_pred = model.predict(X_test)
    fig = px.scatter(x=y_test, y=y_pred, trendline="ols",
                     labels={"x":"Actual","y":"Predicted"},
                     title=f"{best}: Actual vs Predicted")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Ideal predictions should align along the 45‑degree line.")

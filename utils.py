import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc)

NUM_COLS = ["personal_income_usd","online_buy_freq","monthly_spend_usd",
            "social_influence","app_usage","budget_per_box"]
CAT_COLS = ["age_group","gender","region","edu_level","employment",
            "sub_box_past","data_share_ok","push_notif_ok","pay_freq_pref"]

def load_data(source:str="data/cosmetics_survey_synthetic.csv")->pd.DataFrame:
    return pd.read_csv(source)

def basic_preprocess(df:pd.DataFrame, target:str):
    X = df.drop(columns=[target])
    y = df[target]
    numeric_features = [c for c in NUM_COLS if c in X.columns]
    categorical_features = [c for c in CAT_COLS if c in X.columns]
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])
    return X, y, preprocessor

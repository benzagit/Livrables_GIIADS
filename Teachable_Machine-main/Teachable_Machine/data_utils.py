"""
data_utils.py
Fonctions pour le chargement, nettoyage et préparation des données.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_california_housing
import tensorflow as tf

def load_csv(uploaded_file):
    """Charge un fichier CSV ou Excel."""
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Format non supporté. Utilisez CSV ou Excel.")

def load_builtin_dataset(name):
    """Charge un dataset intégré de sklearn."""
    if name == "Iris":
        data = load_iris(as_frame=True)
        df = data.frame
        return df, 'target'
    elif name == "Wine":
        data = load_wine(as_frame=True)
        df = data.frame
        return df, 'target'
    elif name == "Breast Cancer":
        data = load_breast_cancer(as_frame=True)
        df = data.frame
        return df, 'target'
    elif name == "California Housing":
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        return df, 'MedHouseVal'
    elif name == "MNIST (échantillon)":
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_sample = x_train[:1000].reshape(-1, 28*28)
        y_sample = y_train[:1000]
        df = pd.DataFrame(x_sample)
        df['target'] = y_sample
        return df, 'target'
    else:
        raise ValueError("Dataset non reconnu.")

def clean_data(df):
    """Nettoyage basique."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    cols_to_keep = numeric_cols + categorical_cols
    df = df[cols_to_keep].copy()
    df = df.dropna(thresh=int(0.5 * df.shape[1]))
    return df

def preprocess_data(df, target_col, task_type):
    """Prétraitement : encodage, imputation, normalisation."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    class_names = None
    if task_type == "classification":
        if y.dtype == 'object':
            le_y = LabelEncoder()
            y = le_y.fit_transform(y)
            class_names = le_y.classes_.tolist()
        else:
            class_names = sorted(y.unique().astype(str))

    return X.values, y.values, X.columns.tolist(), class_names

def split_data(X, y, test_size=0.2, stratify=False):
    """Split train/test."""
    if stratify and len(np.unique(y)) > 1:
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=42)
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import tensorflow as tf

from data_utils import load_csv, load_builtin_dataset, clean_data, split_data, preprocess_data
from model_utils import get_classical_models, train_classical_model, build_mlp, build_cnn, train_deep_model, save_model, load_trained_model
from evaluation_utils import evaluate_model, plot_confusion_matrix, plot_regression_results, plot_training_curves, plot_roc_curve
from ui_components import upload_data_ui, model_selection_ui, prediction_ui

st.set_page_config(page_title="🎓 Teachable Machine Académique", layout="wide")
st.title("🎓 Teachable Machine Académique")

# === CHOIX INITIAL ===
if 'task_type_forced' not in st.session_state:
    st.session_state.task_type_forced = None

if st.session_state.task_type_forced is None:
    st.sidebar.header("🎯 Étape 1 : Choisissez le type de problème")
    task_choice = st.sidebar.radio("Type de problème", ["Classification", "Régression"], index=0)
    if st.sidebar.button("✅ Valider"):
        st.session_state.task_type_forced = task_choice.lower()
        st.rerun()
else:
    st.sidebar.success(f"Type : {st.session_state.task_type_forced}")
    if st.sidebar.button("🔄 Changer"):
        st.session_state.task_type_forced = None
        st.rerun()

if st.session_state.task_type_forced is None:
    st.info("Choisissez le type de problème dans la barre latérale.")
    st.stop()

# Initialisation
for key in ['data', 'X_raw', 'y_raw', 'task_type', 'model', 'model_type', 'is_trained', 'history', 'class_names', 'feature_names', 'auto_target', 'current_dataset']:
    if key not in st.session_state:
        st.session_state[key] = None

# === CHARGEMENT ===
with st.expander("📂 Données", expanded=True):
    data_source = upload_data_ui()
    if data_source == "Téléverser un fichier":
        uploaded_file = st.file_uploader("CSV/Excel", type=["csv", "xlsx"])
        if uploaded_file:
            try:
                df = load_csv(uploaded_file)
                st.session_state.data = df
                st.session_state.current_dataset = "uploaded"
                st.session_state.auto_target = None

                target_col_guess = df.columns[-1]
                unique_vals = df[target_col_guess].nunique()
                is_cat = df[target_col_guess].dtype in ['object', 'category']
                detected = "classification" if (unique_vals <= 20 and is_cat) else "regression"
                if detected != st.session_state.task_type_forced:
                    st.warning(f"⚠️ Données {detected}, vous avez choisi {st.session_state.task_type_forced}.")
                    if not st.checkbox("✅ Forcer ce type"):
                        st.stop()
                else:
                    st.success(f"✅ Compatible avec {st.session_state.task_type_forced}.")
            except Exception as e:
                st.error(f"Erreur : {e}")
    else:
        CLASSIFICATION_DATASETS = ["Iris", "Wine", "Breast Cancer", "MNIST (échantillon)"]
        REGRESSION_DATASETS = ["California Housing", "Synthetic Regression"]
        datasets = CLASSIFICATION_DATASETS if st.session_state.task_type_forced == "classification" else REGRESSION_DATASETS
        dataset_name = st.selectbox("Dataset", datasets)
        if st.button("Charger"):
            try:
                df, target_col = load_builtin_dataset(dataset_name)
                st.session_state.data = df
                st.session_state.auto_target = target_col
                st.session_state.current_dataset = dataset_name
            except Exception as e:
                st.error(f"Erreur : {e}")

if st.session_state.data is not None:
    df = st.session_state.data
    st.write(f"Shape : {df.shape}")
    st.dataframe(df.head(3))

    with st.expander("⚙️ Préparation", expanded=True):
        target_col = st.selectbox("Cible", df.columns, index=df.columns.get_loc(st.session_state.auto_target) if st.session_state.auto_target in df.columns else 0)
        task_type = st.session_state.task_type_forced
        st.session_state.task_type = task_type

        try:
            df_clean = clean_data(df)
            X_raw, y_raw, feature_names, class_names = preprocess_data(df_clean, target_col, task_type)
            st.session_state.X_raw = X_raw
            st.session_state.y_raw = y_raw
            st.session_state.feature_names = feature_names
            st.session_state.class_names = class_names

            X_train, X_test, y_train, y_test = split_data(X_raw, y_raw, 0.2, stratify=(task_type == "classification"))
            st.session_state.y_train, st.session_state.y_test = y_train, y_test

            # Données pour modèles classiques (toujours 2D)
            st.session_state.X_train_classic = X_train
            st.session_state.X_test_classic = X_test

            # Données pour DL
            if st.session_state.current_dataset == "MNIST (échantillon)" and X_train.shape[1] == 784:
                # Pour CNN : garder 4D
                st.session_state.X_train_cnn = X_train.reshape(-1, 28, 28, 1)
                st.session_state.X_test_cnn = X_test.reshape(-1, 28, 28, 1)
                # Pour MLP : aplatir (déjà 2D, donc identique)
                st.session_state.X_train_mlp = X_train
                st.session_state.X_test_mlp = X_test
            else:
                # Données tabulaires : MLP = classique, CNN non disponible
                st.session_state.X_train_mlp = X_train
                st.session_state.X_test_mlp = X_test
                st.session_state.X_train_cnn = None
                st.session_state.X_test_cnn = None

            st.success("✅ Données prêtes !")
        except Exception as e:
            st.error(f"Préparation échouée : {e}")
            st.stop()

    # === MODÈLE ===
    with st.expander("🧠 Modèle", expanded=True):
        model_category_str = model_selection_ui()
        is_classical = "classiques" in model_category_str
        st.session_state.model_category = "classical" if is_classical else "deep"

        if is_classical:
            models = get_classical_models(task_type)
            model_name = st.selectbox("Modèle", list(models.keys()))
            st.session_state.model_name = model_name
            params = {}
            if "Forest" in model_name:
                params["n_estimators"] = st.slider("Arbres", 10, 200, 100)
            elif "SVM" in model_name or "SVR" in model_name:
                params["C"] = st.slider("C", 0.1, 10.0, 1.0)
            elif "KNN" in model_name:
                params["n_neighbors"] = st.slider("Voisins", 1, 15, 5)
            st.session_state.model_params = params
        else:
            is_image = st.session_state.current_dataset == "MNIST (échantillon)"
            if is_image:
                dl_type = st.radio("Type", ["MLP", "CNN"])
            else:
                st.info("⚠️ Données tabulaires → MLP seulement")
                dl_type = "MLP"
            st.session_state.dl_type = dl_type

            if dl_type == "MLP":
                input_shape = (st.session_state.X_train_mlp.shape[1],)
            else:
                input_shape = st.session_state.X_train_cnn.shape[1:]
            st.session_state.input_shape = input_shape

            if st.radio("Mode", ["Auto", "Custom"]) == "Custom":
                st.session_state.dl_config = {
                    "num_layers": st.slider("Couches", 1, 5, 2),
                    "neurons": st.slider("Neurones", 16, 512, 64),
                    "activation": st.selectbox("Activation", ["relu", "tanh"]),
                    "optimizer": "adam",
                    "epochs": st.slider("Époques", 5, 50, 20),
                    "batch_size": st.selectbox("Batch", [16, 32, 64])
                }
            else:
                st.session_state.dl_config = {}

    # === ENTRAÎNEMENT ===
    if st.button("🚀 Entraîner"):
        try:
            if is_classical:
                # ✅ CORRECTION ICI : ajout de st.session_state.task_type
                model = train_classical_model(
                    model_name,
                    st.session_state.model_params,
                    st.session_state.X_train_classic,
                    st.session_state.y_train,
                    st.session_state.task_type  # ← CETTE LIGNE A ÉTÉ AJOUTÉE
                )
                st.session_state.model = model
                st.session_state.model_type = "classical"
                st.session_state.history = None
                st.session_state.used_X_test = st.session_state.X_test_classic
            else:
                config = st.session_state.dl_config
                n_classes = len(st.session_state.class_names) if task_type == "classification" else 1
                if st.session_state.dl_type == "MLP":
                    model = build_mlp(st.session_state.input_shape, config, task_type, n_classes)
                    X_train_dl = st.session_state.X_train_mlp
                    X_test_dl = st.session_state.X_test_mlp
                else:
                    model = build_cnn(st.session_state.input_shape, config, task_type, n_classes)
                    X_train_dl = st.session_state.X_train_cnn
                    X_test_dl = st.session_state.X_test_cnn

                history = train_deep_model(model, X_train_dl, st.session_state.y_train, X_test_dl, st.session_state.y_test, config)
                st.session_state.model = model
                st.session_state.model_type = "deep"
                st.session_state.history = history
                st.session_state.used_X_test = X_test_dl

            st.session_state.is_trained = True
            st.success("✅ Entraînement terminé !")
        except Exception as e:
            st.error(f"Erreur : {e}")

    # === ÉVALUATION ===
    if st.session_state.is_trained:
        with st.expander("📊 Évaluation", expanded=True):
            X_eval = st.session_state.used_X_test
            y_eval = st.session_state.y_test

            metrics = evaluate_model(
                st.session_state.model,
                X_eval,
                y_eval,
                st.session_state.task_type,
                st.session_state.model_type
            )

            if st.session_state.task_type == "classification":
                st.subheader("📈 Métriques de classification")
                metrics_df = pd.DataFrame({
                    "Accuracy": [metrics["accuracy"]],
                    "Precision (weighted)": [metrics["precision"]],
                    "Recall (weighted)": [metrics["recall"]],
                    "F1-score (weighted)": [metrics["f1"]]
                })
                st.dataframe(metrics_df)

                fig = plot_confusion_matrix(y_eval, metrics["predictions"], st.session_state.class_names)
                st.pyplot(fig)

                if len(st.session_state.class_names) == 2 and "y_proba" in metrics and metrics["y_proba"] is not None:
                    fig = plot_roc_curve(y_eval, metrics["y_proba"])
                    st.pyplot(fig)
            else:
                st.subheader("📈 Métriques de régression")
                st.metric("R²", f"{metrics['r2']:.4f}")
                st.metric("MAE", f"{metrics['mae']:.4f}")
                st.metric("MSE", f"{metrics['mse']:.4f}")
                fig = plot_regression_results(y_eval, metrics["predictions"])
                st.pyplot(fig)

            if st.session_state.history is not None:
                fig = plot_training_curves(st.session_state.history, st.session_state.task_type)
                st.pyplot(fig)

            if st.button("💾 Sauvegarder"):
                save_model(st.session_state.model, "mon_modele", st.session_state.model_type)
                st.success("Sauvegardé dans `models/`")

        if st.session_state.model_type == "classical":
            with st.expander("🔮 Prédiction", expanded=False):
                prediction_ui(st.session_state.model, st.session_state.feature_names, st.session_state.task_type, st.session_state.class_names)

# === RECHARGEMENT ===
st.markdown("---")
st.subheader("📥 Recharger un modèle")
uploaded_model = st.file_uploader("Modèle (.pkl/.h5)", type=["pkl", "h5"])
if uploaded_model:
    try:
        model, model_type = load_trained_model(uploaded_model)
        st.session_state.model = model
        st.session_state.model_type = model_type
        st.session_state.is_trained = True
        st.success("Modèle chargé !")
    except Exception as e:
        st.error(f"Erreur : {e}")
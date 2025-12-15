import streamlit as st

def upload_data_ui():
    return st.radio("Source des données", ["Téléverser un fichier", "Utiliser un dataset intégré"], index=1)

def model_selection_ui():
    return st.radio("Approche", ["🧩 Méthodes classiques (scikit-learn)", "🧠 Deep Learning"])

def prediction_ui(model, feature_names, task_type, class_names=None):
    st.subheader("Entrez les valeurs des features :")
    input_data = []
    for feat in feature_names:
        val = st.number_input(f"{feat}", value=0.0, format="%.4f")
        input_data.append(val)
    
    if st.button("🔍 Prédire"):
        import numpy as np
        X_input = np.array(input_data).reshape(1, -1)
        pred = model.predict(X_input)[0]
        if task_type == "classification":
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_input)[0]
                st.success(f"**Prédiction : {class_names[int(pred)]}**")
                st.write("**Probabilités :**")
                for i, p in enumerate(proba):
                    st.write(f"{class_names[i]}: {p:.2%}")
            else:
                st.success(f"**Prédiction : {class_names[int(pred)]}**")
        else:
            st.success(f"**Valeur prédite : {pred:.4f}**")
import pickle
import os
import numpy as np
import tensorflow as tf
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def get_classical_models(task_type):
    if task_type == "classification":
        return {
            "Logistic Regression": LogisticRegression,
            "Decision Tree": DecisionTreeClassifier,
            "Random Forest": RandomForestClassifier,
            "SVM": SVC,
            "KNN": KNeighborsClassifier,
            "Naive Bayes": GaussianNB,
            "Gradient Boosting": GradientBoostingClassifier
        }
    else:
        return {
            "Linear Regression": LinearRegression,
            "Ridge": Ridge,
            "Lasso": Lasso,
            "Decision Tree": DecisionTreeRegressor,
            "Random Forest": RandomForestRegressor,
            "SVR": SVR,
            "KNN": KNeighborsRegressor,
            "Gradient Boosting": GradientBoostingRegressor
        }

def train_classical_model(model_name, params, X_train, y_train, task_type):
    """CORRIGÉ : utilise le task_type imposé par l'utilisateur."""
    models = get_classical_models(task_type)  # ← ICI : on respecte le choix utilisateur
    model_class = models[model_name]
    
    import inspect
    sig = inspect.signature(model_class.__init__)
    allowed_params = set(sig.parameters.keys()) - {'self'}
    filtered_params = {k: v for k, v in params.items() if k in allowed_params}
    
    model = model_class(**filtered_params)
    model.fit(X_train, y_train)
    return model

def build_mlp(input_shape, config, task_type, n_classes=1):
    model = Sequential()
    model.add(Dense(config.get("neurons", 64), activation=config.get("activation", "relu"), input_shape=input_shape))
    for _ in range(config.get("num_layers", 2) - 1):
        model.add(Dense(config.get("neurons", 64), activation=config.get("activation", "relu")))
    if task_type == "classification":
        model.add(Dense(n_classes, activation="softmax" if n_classes > 2 else "sigmoid"))
        model.compile(optimizer=config.get("optimizer", "adam"), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(1))
        model.compile(optimizer=config.get("optimizer", "adam"), loss="mse", metrics=["mae"])
    return model

def build_cnn(input_shape, config, task_type, n_classes=1):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten())
    model.add(Dense(config.get("neurons", 64), activation='relu'))
    if task_type == "classification":
        model.add(Dense(n_classes, activation="softmax" if n_classes > 2 else "sigmoid"))
        model.compile(optimizer=config.get("optimizer", "adam"), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    else:
        model.add(Dense(1))
        model.compile(optimizer=config.get("optimizer", "adam"), loss="mse", metrics=["mae"])
    return model

def train_deep_model(model, X_train, y_train, X_val, y_val, config):
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 32)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    return history

def save_model(model, name, model_type):
    path = os.path.join(MODEL_DIR, name)
    if model_type == "classical":
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(model, f)
    else:
        model.save(f"{path}.h5")

def load_trained_model(uploaded_file):
    file_name = uploaded_file.name
    if file_name.endswith(".pkl"):
        model = pickle.load(uploaded_file)
        return model, "classical"
    elif file_name.endswith(".h5"):
        model = tf.keras.models.load_model(uploaded_file)
        return model, "deep"
    else:
        raise ValueError("Format de modèle non supporté.")
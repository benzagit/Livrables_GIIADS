# 🎓 Teachable Machine — Academic Edition

A minimal, educational Streamlit app to **build, train, evaluate, and deploy** Machine Learning and Deep Learning models for **image and tabular data** — all in your browser.

---

## ✨ Features

- **Task Selection**: Choose **Classification** or **Regression** upfront.
- **Model Types**:
  - 🧩 **Classical ML**: Scikit-learn (Random Forest, SVM, Logistic Regression, etc.)
  - 🧠 **Deep Learning**: Custom MLP or CNN (Keras/TensorFlow)
- **Built-in Datasets**:
  - Classification: `Iris`, `Wine`, `Breast Cancer`, `MNIST`
  - Regression: `California Housing`, `Synthetic Regression`
- **Full Evaluation**:
  - Classification: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC Curve
  - Regression: R², MAE, MSE, Prediction Plots
- **Real-time Prediction**: Manual input or image upload
- **Model Persistence**: Save/load `.pkl` (sklearn) or `.h5` (Keras)

---

## 🚀 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/teachable-machine.git
   cd teachable-machine
   ```

2. **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```

3. **Run the app**
  ```bash
  streamlit run main_app.py
  ```

## 📁 Project Structure

  ```bash
  ├── main_app.py             # Streamlit entry point
  ├── data_utils.py           # Data loading & preprocessing
  ├── model_utils.py          # Model building & training
  ├── evaluation_utils.py     # Metrics & visualizations
  ├── ui_components.py        # Reusable UI elements
  ├── requirements.txt        # Dependencies
  └── .gitignore              # Excludes models/, cache, etc.
  ```




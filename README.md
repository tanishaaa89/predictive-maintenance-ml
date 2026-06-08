# ⚙️ Predictive Maintenance — Machine Failure Prediction

An end-to-end machine learning project that predicts industrial machine 
failure using real-time sensor data, deployed as an interactive web app.

## 🚀 Live Demo
[**Try the app here →**](https://predictive-maintenance-tanisha.streamlit.app)

## 📊 Project Overview
Built on the AI4I 2020 Predictive Maintenance Dataset (10,000 sensor 
readings), this project predicts whether a machine is at risk of failure 
based on temperature, torque, rotational speed, and tool wear readings.

## 🔍 Key Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 85.95% | 0.937 |
| Random Forest | 98.90% | 0.961 |
| **Gradient Boosting ✅** | **99.25%** | **0.967** |
| SVM | 92.35% | 0.972 |

**Selected model: Gradient Boosting**
- Caught 54 out of 68 actual failures in test set
- Only 1 false alarm out of 1,932 normal readings
- All 3 engineered features ranked in top 4 predictors

## 🛠️ Feature Engineering
Three domain-knowledge features engineered from raw sensor data:

| Feature | Formula | Insight |
|---------|---------|---------|
| Power | Torque × Rotational Speed | Mechanical load proxy |
| Tool_wear_torque | Tool Wear × Torque | Combined stress indicator |
| Temp_difference | Process Temp − Air Temp | Thermal stress indicator |

These engineered features outperformed all raw sensor readings in 
importance scoring.

## 🧠 Tech Stack
Python · Scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · 
Streamlit · Joblib

## 📁 Project Structure
├── app.py                  # Streamlit web application
├── model.pkl               # Trained Gradient Boosting model
├── scaler.pkl              # StandardScaler for preprocessing
├── feature_names.pkl       # Feature column names
├── requirements.txt        # Dependencies

## 📂 Dataset
AI4I 2020 Predictive Maintenance Dataset  
10,000 rows · 14 features · 3.39% failure rate  
Source: [Kaggle](https://kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020)

## 👩‍💻 Author
**Tanisha Sharma** — B.Tech CSE, VIPS-TC Delhi  

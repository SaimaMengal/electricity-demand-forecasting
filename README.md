# ⚡ Electricity Demand Forecasting

## 📌 Project Overview
This project analyzes 12 years (2006-2017) of real electricity 
consumption data from Germany to build accurate demand 
forecasting models using Machine Learning and Deep Learning.

## 🏆 Model Results
| Model | MAE | Accuracy |
|-------|-----|----------|
| Random Forest (Basic) | 33.19 GWh | 97.5% |
| Random Forest (Lag Features) | 24.96 GWh | 98.1% |
| LSTM Deep Learning | 32.55 GWh | 97.6% |

## 🛠️ Technologies Used
- **Python 3.12**
- **Pandas & NumPy** — Data processing
- **Scikit-learn** — Random Forest model
- **TensorFlow/Keras** — LSTM Deep Learning model
- **Matplotlib** — Data visualization

## 📊 Key Findings
- Day of Week is the most important feature (54% importance)
- Adding Lag features improved accuracy by 25%
- Random Forest outperformed LSTM on this dataset size
- Clear seasonal patterns found — higher consumption in winter

## 📁 Dataset
- **Source:** Open Power System Data (Germany)
- **Duration:** 2006 to 2017 (4,383 days)
- **Columns:** Date, Consumption, Wind, Solar, Wind+Solar

## 🔄 Project Workflow

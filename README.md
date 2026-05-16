# ⚡ Electricity Demand Forecasting

## 📌 Project Overview
Germany ka 12 saal ka (2006-2017) real electricity data 
use karke demand forecasting models banaye gaye.

## 🏆 Results
| Model | MAE | Accuracy |
|-------|-----|----------|
| Random Forest Basic | 33.19 GWh | 97.5% |
| Random Forest + Lag | 24.96 GWh | 98.1% |
| LSTM Deep Learning | 32.55 GWh | 97.6% |

## 🛠️ Technologies Used
- Python 3.12
- Pandas, NumPy
- Scikit-learn (Random Forest)
- TensorFlow/Keras (LSTM)
- Matplotlib

## 📊 Key Findings
- DayOfWeek sabse important feature (54% importance)
- Lag features ne accuracy 25% improve ki
- Random Forest small dataset par LSTM se behtar raha

## 📁 Dataset
- Source: Open Power System Data (Germany)
- Duration: 2006-2017 (4383 days)
- Features: Consumption, Wind, Solar

## 👩‍💻 Author
Saima Mengal — AI & Machine Learning Enthusiast

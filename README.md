# 💡 SmartPremium: Insurance Premium Prediction

## 📌 Project Overview
SmartPremium is a machine learning project that predicts **insurance premiums** based on customer information such as age, income, health score, claims history, vehicle details, and property type.  

The workflow includes:
- Data preprocessing & feature engineering  
- Model training & evaluation  
- Saving the best model  
- Making predictions on test data  
- Deploying a **Streamlit app** for user interaction  

---

## 📂 Project Structure
```plaintext
SmartPremium/
│── train.csv              # Training dataset  
│── test.csv               # Test dataset  
│── Smart_premium.ipynb    # Notebook for training & evaluation  
│── Test_prediction.ipynb  # Notebook for generating predictions  
│── best_model.pkl         # Trained model saved using joblib  
│── submission.csv         # Final prediction results  
│── app.py                 # Streamlit app for deployment  
```

## ⚙️ Setup Instructions
### 1. Clone the repository
```bash
git clone <your_repo_link>
cd SmartPremium
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3.Install dependencies
```bash
pip install pandas numpy scikit-learn joblib streamlit
pip install matplotlib seaborn plotly
```

## 📊 Model Training & Evaluation
1. Open Smart_premium.ipynb.
2. Load train.csv and perform feature engineering.
3. Train models and evaluate using:
    -`RMSE (Root Mean Squared Error)`
    -`MAE (Mean Absolute Error)`
    -`R² Score`
4. Save the final trained model as best_model.pkl.

## 🔮 Generating Predictions
1. Open `Test_prediction.ipynb`.
2. Load `test.csv` and `best_model.pkl`.
3. Apply the same preprocessing steps.
4. Generate predictions.
5. Save them as `submission.csv`.

## 🌐 Running the Streamlit App
Run the app with:
```bash
streamlit run app.py
```
This will launch the app at http://localhost:8501

### Features:
- User-Friendly form to input customer details.
- Real-time premium prediction 💰.

## 🏆 Deliverables
- `best_model.pkl` --> Trained ML model
- `submission.csv` --> Prediction on test data
- **Streamlit app** (`app.py`) --> Intercation premium predictoe
# Loan Repayment Prediction with Deep Learning

## 📌 Overview
This project uses a **deep learning model (Keras/TensorFlow)** to predict whether a borrower will repay a loan, based on historical LendingClub loan data.  
The goal is to help financial institutions identify **high-risk borrowers** and minimize defaults.

## 📊 Dataset
- **Source**: [LendingClub Dataset on Kaggle](https://www.kaggle.com/wordsforthewise/lending-club)  
- **Target Variable**: `loan_status` → indicates if the borrower paid back the loan or defaulted.  
- **Features**: Loan amount, interest rate, installment, grade, employment length, home ownership, annual income, DTI (debt-to-income ratio), credit history, etc.  

## ⚙️ Steps in the Notebook
1. **Data Exploration**  
   - Load and inspect LendingClub dataset  
   - Handle missing values and outliers  
   - Visualize key features (loan amount, term, grade, purpose, etc.)  

2. **Feature Engineering & Preprocessing**  
   - Convert categorical features to dummy variables  
   - Standardize/scale numerical features  
   - Train-test split  

3. **Model Development**  
   - Neural network built with **Keras Sequential API**  
   - Architecture: Dense layers with ReLU activations + dropout for regularization  
   - Optimizer: Adam  
   - Loss: Binary cross-entropy  

4. **Training & Evaluation**  
   - Model trained over multiple epochs with batch processing  
   - Evaluation using accuracy, classification report, and confusion matrix  
   - Visualization of training/validation loss and accuracy  

5. **Predictions**  
   - Predict loan repayment likelihood on test data  
   - Interpret results to identify high-risk loans  

## 🏗️ Tech Stack
- **Python**  
- **Pandas / NumPy**  
- **Matplotlib / Seaborn**  
- **Scikit-learn**  
- **TensorFlow / Keras**

## 📈 Results
- Built a binary classification deep learning model for loan repayment prediction  
- Demonstrated performance improvements through feature engineering and model tuning  
- Visualizations provided insights into borrower behavior and repayment patterns  

## 📂 Repository Structure
- Loan_Repayment_Prediction_Deep_Learning.ipynb # Main notebook
- lending_club_loan_two.csv # Dataset
- lending_club_info.csv # Feature information
- README.md # Project documentation

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/neerajingle95/Loan-Repayment-Prediction-with-Deep-Learning.git

2. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook Loan_Repayment_Prediction_Deep_Learning.ipynb

## 🙌 Acknowledgements
- Dataset: LendingClub on Kaggle
- Libraries: TensorFlow, Keras, Scikit-learn, Pandas, Matplotlib, Seaborn

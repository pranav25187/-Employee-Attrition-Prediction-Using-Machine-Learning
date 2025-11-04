# 🧠 Employee Attrition Prediction Using Machine Learning

### 👋 Hello Everyone!
I’m **Pranav Balkrishna Borse**, and this is my **Employee Attrition Prediction** project built using **Machine Learning**.  
This project aims to predict whether an employee will leave an organization based on multiple factors, using various supervised ML models and performance evaluation metrics.

---

## 🎯 Objective
Employee attrition refers to an employee’s voluntary or involuntary resignation from an organization.  
Since hiring and training employees require significant investment, retaining skilled employees is crucial for business success.  

The goal of this project is to:
- Predict employee attrition using machine learning algorithms.  
- Identify key factors that contribute to employee turnover.  
- Provide insights that can help organizations improve employee retention.

---

## ⚙️ Methodology

### 🔄 Workflow
The project follows the standard **Machine Learning Pipeline**:
1. Data Collection and Preprocessing  
2. Exploratory Data Analysis (EDA)  
3. Feature Engineering and Transformation  
4. Model Training and Evaluation  
5. Model Interpretation and Insights  

---

## 🤖 Machine Learning Models Used
We trained and evaluated **9 supervised classification models** to predict employee attrition:

- Logistic Regression  
- Naive Bayes  
- Decision Tree  
- Random Forest  
- AdaBoost  
- Support Vector Machine (SVM)  
- Linear Discriminant Analysis (LDA)  
- Multilayer Perceptron (MLP)  
- K-Nearest Neighbors (KNN)

---

## 🧩 Datasets Used
We experimented with **6 different dataset versions** to handle imbalance and dimensionality reduction:

1. Imbalanced Data  
2. Undersampled Data  
3. Oversampled Data  
4. PCA (Principal Component Analysis)  
5. Undersampling with PCA  
6. Oversampling with PCA  

For optimal performance:
- **Hyperparameter tuning** was done using `RandomSearchCV` and `GridSearchCV`.  
- **5-Fold Cross Validation** was used to ensure model robustness.  
- Models were evaluated on metrics such as **Accuracy**, **Precision**, **Recall**, and **F1 Score**.

---

## 📊 Dataset Details
The dataset used is the **IBM Employee Attrition Dataset** from Kaggle.  
It contains:
- **1470 rows** and **35 features**  
- A mix of **numerical and categorical variables**

Each record represents an employee’s demographic, job satisfaction, and performance data.

---

## 🧠 Results and Insights
The **Random Forest Model** with **PCA and Oversampling** achieved the best results:

| Metric | Score |
|---------|--------|
| Accuracy | 99.2% |
| Precision | 98.6% |
| Recall | 99.8% |
| F1 Score | 99.2% |

### 🔍 Key Influential Features
- **Most Important:** MonthlyIncome, OverTime, Age  
- **Least Important:** Performance Rating, Gender, Business Travel  

---

## 🚀 How to Run

You can run this project using **Google Colab** or **Anaconda Jupyter Notebook**.

### ▶️ Steps to Run (Google Colab)
1. Upload the dataset.  
2. Open the Jupyter Notebook file.  
3. Go to **Runtime → Run all** or **Restart and Run all**.  

---

## 📚 Libraries Used
- **NumPy**  
- **Pandas**  
- **Matplotlib**  
- **Seaborn**  
- **Scikit-learn**

---

## 📂 Dataset Source
You can access the dataset here:  
🔗 [IBM HR Analytics Employee Attrition Dataset – Kaggle](https://www.kaggle.com/)

---

### 👨‍💻 Author
**Pranav Balkrishna Borse**  
📧 *pranavb2506@gmail.com*  
🔗 GitHub: [github.com/pranav-borse](https://github.com/pranav25187)  


# Customer Segmentation System

A full-stack Machine Learning web application that segments customers based on income and spending behavior using K-Means Clustering.

The application helps businesses identify different customer groups such as premium customers, budget customers, and regular shoppers.

---

# Features

* Customer segmentation using Machine Learning
* Customer prediction system
* Interactive dashboard
* Analytics and visualizations
* SQLite database integration
* Prediction history tracking
* Responsive UI with Streamlit

---

# Technologies Used

| Area             | Technology         |
| ---------------- | ------------------ |
| Language         | Python             |
| Machine Learning | Scikit-learn       |
| Frontend         | Streamlit          |
| Database         | SQLite             |
| Visualization    | Matplotlib, Plotly |

---

# Machine Learning Model

The project uses:

K-Means Clustering


to group customers based on:

* Annual Income
* Spending Score

---

# Project Pipeline


Dataset Collection
        ↓
Data Preprocessing
        ↓
Feature Scaling
        ↓
K-Means Clustering
        ↓
Customer Prediction
        ↓
Visualization
        ↓
Database Storage
        ↓
Deployment
```

---

# Dataset

Dataset used:


Mall Customer Segmentation Dataset


Source:

[Kaggle Dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python?utm_source=chatgpt.com)

---

# Project Structure

customer-segmentation-system/
│
├── app.py
├── database.py
├── customers.db
├── mall_customers.csv
├── requirements.txt
├── README.md


---

# How to Run

Install dependencies:


pip install -r requirements.txt


Run the app:


streamlit run app.py


---

# Application Pages

* Dashboard
* Prediction
* History
* Analytics

---

# Future Improvements

* User authentication
* Cloud database
* Advanced ML models
* Report generation

---

# Conclusion

This project demonstrates the integration of Machine Learning, Full Stack Development, Database Management, and Data Visualization in a real-world business application.

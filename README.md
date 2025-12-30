# Phishing URL Detection System 🛡️

## 📌 Project Overview
Phishing attacks are one of the most common security threats today. This project utilizes **Machine Learning** and **Natural Language Processing (NLP)** techniques to detect malicious URLs. By analyzing the lexical features of a URL, the model can predict whether a link is safe or a potential phishing attempt.

## 🚀 Features
* **Data Preprocessing:** Cleaning and tokenizing URL strings.
* **Feature Extraction:** Using **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert URL text into numerical vectors.
* **Model Comparison:** Training and evaluating multiple algorithms to find the best accuracy.
* **Prediction:** Classifying URLs as either `Legitimate` or `Phishing`.

## 🛠️ Technologies Used
* **Language:** Python 🐍
* **Libraries:**
    * `pandas` & `numpy` (Data Manipulation)
    * `matplotlib` & `seaborn` (Data Visualization)
    * `scikit-learn` (Machine Learning & Metrics)
* **Environment:** Google Colab / Jupyter Notebook

## 🤖 Algorithms Implemented
The following algorithms were explored and compared in this project:
1.  **Logistic Regression** (Baseline model)
2.  **Multinomial Naive Bayes** (Great for text classification)
3.  **Random Forest Classifier** (Ensemble method)
4.  **XGBoost** (Gradient Boosting for high performance)

## 📊 Results
After training the models, the following accuracy scores were observed:

| Model | Accuracy |
| :--- | :--- |
| Logistic Regression | [e.g., 85%] |
| Naive Bayes | [e.g., 88%] |
| Random Forest | [e.g., 94%] |
| **XGBoost** | **[e.g., 96%]** |

*The best performing model was **[Insert Best Model Name]** with an accuracy of **[Insert %]**.*

## 📂 Dataset
* The dataset contains URLs labeled as either legitimate or phishing.
* *Source:* [Mention where you got the data, e.g., Kaggle, or "Open Source Phishing Database"]

## ⚙️ How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/Phishing-URL-Detection.git](https://github.com/YourUsername/Phishing-URL-Detection.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn seaborn xgboost
    ```
3.  Open the Jupyter Notebook (`.ipynb`) and run the cells.

## 🤝 Contribution
Feel free to fork this repository and submit pull requests to improve the feature extraction or model performance.


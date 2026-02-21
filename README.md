# 🏠 California Housing Machine Learning Analytics Dashboard

---

## 📘 Course Information

- **Course Code:** CS33002  
- **Course Name:** Applications Development Laboratory  
- **Student Name:** [Your Name]  
- **Roll Number:** [Your Roll Number]  
- **Semester:** 6th  
- **Instructor:** [Instructor Name]  

---

## 📌 Project Overview

This project implements a complete **end-to-end Machine Learning pipeline** using the California Housing dataset from:


The system includes data preprocessing, multiple ML models, evaluation, clustering, neural networks, and final deployment as a modern Flask-based web application.

The final web dashboard allows users to:

- Predict Median House Value (Regression)
- Predict Housing Category (Classification)
- Identify Housing Region Cluster (Clustering)

The project strictly follows the fixed data split and dataset constraints defined in the assignment guidelines.

---

# 🎯 Learning Tasks

From the same dataset, three distinct machine learning problems were defined:

---

## 1️⃣ Regression Task

**Objective:**  
Predict the continuous target variable:

- `MedHouseVal` (Median House Value)

**Models Trained:**
- Simple Linear Regression
- Multiple Linear Regression (Final Selected Model)

**Evaluation Metrics:**
- Mean Squared Error (MSE)
- R² Score
- Actual vs Predicted Plot

---

## 2️⃣ Classification Task (Derived)

The continuous target was converted into three classes:

- **Low Value** → Bottom 33%
- **Medium Value** → Middle 33%
- **High Value** → Top 33%

**Models Trained:**
- Logistic Regression
- Decision Tree
- Random Forest (Selected Model)
- Support Vector Machine
- Neural Network (MLP)

**Evaluation Metrics:**
- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score

---

## 3️⃣ Clustering Task

Grouped housing regions based on socio-economic and geographical features.

**Model Used:**
- KMeans Clustering

**Evaluation Methods:**
- Elbow Method
- Silhouette Score
- 2D & 3D PCA Visualization

---

# 📊 Fixed Data Split (Mandatory Rule)

To ensure fairness and reproducibility:

- **Training Set:** 70%
- **Validation Set:** 15%
- **Testing Set:** 15%
- **random_state = 42**

Important:
- Validation and Test sets were never used during preprocessing.
- Feature scaling was performed using only training data statistics.

---

# 🧩 Project Phases

---

## 🔹 Phase 1: Data Preprocessing & EDA

- Dataset loading
- Train/Validation/Test split (70/15/15)
- Feature scaling
- Missing value check
- Visualizations:
  - Histogram
  - Scatter plot
  - Correlation heatmap

---

## 🔹 Phase 2: Regression Analysis

- Simple Linear Regression
- Multiple Linear Regression
- Validation tuning
- Final evaluation on test set

---

## 🔹 Phase 3: Classification Models

- Logistic Regression
- Decision Tree
- Random Forest
- Model comparison using validation set
- Final test evaluation

---

## 🔹 Phase 4: Support Vector Machine

- Linear & RBF kernels tested
- Hyperparameter tuning using validation set
- Test evaluation
- Comparison with Random Forest

---

## 🔹 Phase 5: Neural Network

- Multi-Layer Perceptron (MLPClassifier)
- Early stopping using validation set
- Training vs Validation accuracy plot
- Training vs Validation loss plot
- Final test evaluation

---

## 🔹 Phase 6: Web Deployment

Backend:
- Flask
- Model loading using joblib
- Real-time prediction API

Frontend:
- Modern dashboard UI
- Glassmorphism effects
- Animated prediction output
- Responsive Bootstrap design

---

# 🖥️ Web Application Features

The web dashboard contains:

### 🏠 Home Page
- Dataset explanation
- Phase overview
- Learning task summary
- Data split visualization

### 📊 Reports Page
- EDA graphs
- Regression evaluation plots
- Neural network training curves
- PCA and clustering visualizations

### 🔮 Prediction Page
- Input housing features
- Outputs:
  - Predicted House Price
  - Predicted Category (Low / Medium / High)
  - Cluster Group

---

# 📁 Project Structure

housing_project/
│
├── notebook/
│ └── housing.ipynb
│
├── models/
│ ├── regression_model.pkl
│ ├── random_forest_model.pkl
│ ├── svm_model.pkl
│ ├── neural_network_model.pkl
│ ├── kmeans_model.pkl
│ ├── scaler.pkl
│ └── cluster_scaler.pkl
│
├── static/
│ ├── style/
│   |── home.css
│   └── predict.css
│ └── images/
│ └── (saved plots)
│
├── templates/
│ ├── home.html
│ └── predict.html
│
├── app.py
├── requirements.txt
└── README.md


---

# ⚙️ Installation & Setup

## 1️ Create Virtual Environment

```bash
python -m venv venv
## 2️ Activate Environment
Windows:
venv\Scripts\activate

## 3️ Install Dependencies
pip install -r requirements.txt

## 4️ Run the Flask Application
python app.py

Open your browser and visit:

http://127.0.0.1:5000


### 🧪 Example Input
Feature	Example Value
MedInc	    3.5
HouseAge	25
AveRooms	5.5
AveBedrms	1
Population	1200
AveOccup	3
Latitude	34.2
Longitude	-118.4

### 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Flask
- HTML5
- CSS3
- Bootstrap 5
- Joblib

📊 Dataset

California Housing Dataset
Source: sklearn.datasets.fetch_california_housing()

The dataset contains housing data from the 1990 California census and includes socio-economic and geographic attributes.

🚀 Project Status

✔ Data preprocessing completed
✔ Regression models trained and evaluated
✔ Classification models compared
✔ SVM and Neural Network implemented
✔ Clustering with PCA visualization
✔ Models saved using pickle
✔ Web deployment completed

📄 Academic Declaration

This project was developed strictly according to the assignment requirements.
Only the California Housing dataset was used, and all data splitting rules were followed as specified.
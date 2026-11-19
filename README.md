# Student Academic Performance Prediction Using Deep Neural Networks

A machine learning project that predicts students' academic performance using the UCI Student Performance dataset. The project implements a Deep Neural Network (DNN), compares it against Logistic Regression, and provides an interactive Streamlit application.

## Project Overview

The objective is to classify students into four academic performance categories based on demographic, family, academic, and lifestyle information.

Two prediction scenarios are implemented:

- **Early Prediction:** Predicts student performance without using previous period grades (G1 and G2).
- **Late Prediction:** Predicts student performance using previous period grades (G1 and G2).

## Dataset

- **Source:** UCI Student Performance Dataset
- **Dataset:** Portuguese language student records (`student-por.csv`)
- **Total Records:** 649
- **Original Features:** 32 input features
- **Target:** Final grade (G3)

### Performance Categories

| Final Grade (G3) | Category |
|---|---|
| 0–9 | Poor |
| 10–13 | Average |
| 14–16 | Good |
| 17–20 | Excellent |

## Data Preprocessing

- Separated numerical and categorical features.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Encoded target classes.
- Created separate early and late prediction datasets.
- Used stratified train-validation-test splitting.

### Dataset Split

| Split | Records | Percentage |
|---|---:|---:|
| Training | 454 | 70% |
| Validation | 97 | 15% |
| Testing | 98 | 15% |

The same student records were used across both prediction scenarios to ensure a fair comparison.

## Models

### 1. Logistic Regression

Used as the traditional machine learning baseline for both prediction scenarios.

### 2. Deep Neural Network

Architecture:

- Input Layer
- Dense Layer: 64 neurons, ReLU
- Dropout: 0.3
- Dense Layer: 32 neurons, ReLU
- Dropout: 0.2
- Output Layer: 4 neurons, Softmax

**Training Configuration:**

- Optimizer: Adam Optimizer
- Learning Rate: 0.001
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 16
- Maximum Epochs: 100
- Early Stopping: Patience of 10 epochs

## Model Evaluation

Models were evaluated on the independent test dataset using Accuracy and Macro F1.

| Model | Scenario | Accuracy | Macro F1 |
|---|---|---:|---:|
| Logistic Regression | Early | 53.06% | 0.3596 |
| Logistic Regression | Late | 76.53% | 0.6790 |
| DNN | Early | 52.04% | 0.2977 |
| DNN | Late | 72.45% | 0.6623 |

### Key Observations

- Late prediction achieved better performance than early prediction for both models.
- Previous academic grades (G1 and G2) provide useful predictive information.
- Logistic Regression outperformed the DNN in both test scenarios.
- The results demonstrate that a simpler machine learning model can outperform a neural network on a relatively small dataset.

## Streamlit Application

The project includes an interactive Streamlit application that allows users to:

- Select early or late prediction.
- Enter student demographic and academic information.
- Generate academic performance predictions.
- View one of four predicted categories: Poor, Average, Good, or Excellent.

## Project Structure

```text
Student-Performance-DNN/
│
├── data/
│   └── student-por.csv
│
├── models/
│   ├── dnn_early.keras
│   ├── dnn_late.keras
│   ├── preprocessor_early.pkl
│   ├── preprocessor_late.pkl
│   ├── label_encoder.pkl
│   └── final_model_comparison.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_logistic_regression.ipynb
│   ├── 04_dnn_model.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_hyperparameter_tuning.ipynb
│   └── 07_final_evaluation.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/theparvmahajan/student-performance-dnn.git
cd student-performance-dnn
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Jupyter Notebook

## Limitations

- The dataset represents students from Portuguese schools and may not generalize to other educational systems.
- Predictions are estimates and should not be treated as definitive assessments of student ability.

## Author

**Parv Mahajan**

GitHub: https://github.com/theparvmahajan


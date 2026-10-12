# 🎓 Student Performance Prediction Using a Decision Tree Classifier

### Fundamentals of Artificial Intelligence (FAI) — Case Study Project

<p align="center">
  A machine learning application that predicts student performance categories using academic and behavioral indicators.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/Algorithm-Decision%20Tree-green" alt="Decision Tree">
  <img src="https://img.shields.io/badge/Framework-Streamlit-FF4B4B?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Project status">
</p>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Objectives](#-objectives)
- [Key Features](#-key-features)
- [Machine Learning Algorithm](#-machine-learning-algorithm)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Dataset Description](#-dataset-description)
- [Project Structure](#-project-structure)
- [Model Evaluation](#-model-evaluation)
- [Application Screenshots](#-application-screenshots)
- [Installation and Setup](#-installation-and-setup)
- [How to Use the Application](#-how-to-use-the-application)
- [Sample Output](#-sample-output)
- [Limitations](#-limitations)
- [Future Enhancements](#-future-enhancements)
- [Conclusion](#-conclusion)
- [Academic Disclaimer](#-academic-disclaimer)
- [Author and Repository](#-author-and-repository)

---

## 📖 Project Overview

Student Performance Prediction Using a Decision Tree Classifier is a machine learning project developed for the Fundamentals of Artificial Intelligence (FAI) case study.

The project demonstrates how supervised machine learning can classify student academic performance using selected academic and behavioral indicators. It uses a Decision Tree Classifier implemented with Scikit-learn and provides an interactive web interface developed with Streamlit.

Users can enter a student's study hours, attendance percentage, previous examination score, and assignment completion percentage. The trained model predicts one of three performance categories:

- **Low Performance**
- **Medium Performance**
- **High Performance**

The application also displays prediction confidence across the three classes and provides general educational recommendations based on the predicted category.

The project demonstrates the complete machine learning workflow, including synthetic data generation, model training, testing, evaluation, serialization, and interactive prediction.

## 🎯 Problem Statement

Academic performance can be influenced by several factors, including study habits, attendance, previous examination results, and assignment completion.

Understanding how these indicators relate to performance can help demonstrate the potential of machine learning for educational analysis. However, manually evaluating multiple indicators can be difficult when dealing with many students.

This project implements a Decision Tree classification model to demonstrate how these features can be used to predict performance categories through a simple and interpretable machine learning pipeline.

The current implementation uses synthetic demonstration data. It is intended to illustrate the approach rather than make real academic assessments.

## 🎯 Objectives

1. Understand and implement a supervised machine learning classification algorithm.
2. Generate a reproducible dataset for student performance prediction.
3. Train a Decision Tree Classifier using Scikit-learn.
4. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
5. Save the trained model and its metadata for reuse.
6. Develop an interactive Streamlit application for real-time predictions.
7. Demonstrate the complete workflow from model training to user-facing inference.
8. Document the project so that others can reproduce the results.

## ✨ Key Features

- **Machine Learning Classification:** Predicts Low, Medium, or High performance.
- **Four Input Features:** Study hours, attendance, previous exam score, and assignment completion.
- **Interactive Dashboard:** Uses Streamlit sliders to enter student information.
- **Prediction Confidence:** Displays the model's class probability estimates.
- **Model Evaluation:** Reports test accuracy, precision, recall, F1-score, and confusion matrix.
- **Reproducible Training:** Uses fixed random seeds for repeatable dataset generation and model training.
- **Model Persistence:** Saves the trained model using Joblib.
- **Educational Recommendations:** Provides general suggestions associated with each predicted category.
- **Explainability:** Introduces decision rules, hierarchical splitting, and Gini impurity.
- **Reproducible Setup:** Includes requirements and instructions to regenerate the model locally.

## 🤖 Machine Learning Algorithm

### Decision Tree Classifier

A Decision Tree is a supervised machine learning algorithm that learns decision rules from labeled training data to predict a target class.

The model divides data into smaller groups using feature-based conditions. Each internal node represents a decision, each branch represents an outcome, and each leaf node represents a predicted class.

### Why was a Decision Tree selected?

- **Interpretability:** Its hierarchical decision structure is relatively easy to explain.
- **Classification:** It supports prediction across multiple discrete classes.
- **Efficiency:** It can train and predict on a small tabular dataset without a GPU.
- **Educational value:** It helps demonstrate training, testing, evaluation, and model inference.

### How the model works

1. Generate the dataset containing input features and performance labels.
2. Separate the features from the target labels.
3. Split the dataset into training and testing subsets.
4. Train the Decision Tree using the training subset.
5. Evaluate predictions on unseen test samples.
6. Save the trained model and metadata.
7. Load the saved model in Streamlit to classify new user inputs.

### Model configuration

The training script uses a `DecisionTreeClassifier` with the following configuration:

| Parameter | Value |
|---|---|
| Criterion | Gini impurity |
| Maximum depth | 4 |
| Minimum samples to split | 5 |
| Minimum samples per leaf | 3 |
| Random state | 42 |
| Training split | 80% |
| Testing split | 20% |

The maximum tree depth and minimum sample constraints limit tree complexity and help make the learned decision structure easier to interpret.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Numerical operations and reproducible synthetic data generation |
| pandas | Dataset creation and tabular data handling |
| Scikit-learn | Decision Tree training, splitting, and evaluation |
| Joblib | Saving and loading the trained model |
| Streamlit | Interactive web application |
| Matplotlib | Supporting visualization dependency |
| Git | Version control |
| GitHub | Source code hosting and project sharing |

## 🏗️ System Architecture

The project follows a straightforward machine learning pipeline.

```text
Synthetic Data Generation
          |
          v
Dataset Creation (500 Records)
          |
          v
Feature and Target Separation
          |
          v
Train-Test Split (80:20)
          |
          v
Decision Tree Classifier
          |
          v
Model Evaluation
          |
          v
Save Model and Metadata
          |
          v
Streamlit Web Application
          |
          v
User Inputs and Feature Validation
          |
          v
Performance Prediction
          |
          v
Prediction Confidence and Guidance
```

The model is trained separately using `train_model.py`. The Streamlit application loads the saved model rather than retraining it whenever a user makes a prediction.

## 📊 Dataset Description

### Dataset generation

The project generates a synthetic dataset containing **500 student records** using NumPy and a fixed random seed of `42`.

The dataset is designed for demonstration and reproducibility. It does not contain real student records.

### Input features

| Feature | Description | Input range |
|---|---|---|
| `study_hours` | Daily study duration in hours | 1.0–9.5 |
| `attendance_percentage` | Percentage of classes attended | 50–100 |
| `previous_exam_score` | Previous examination score | 40–98 |
| `assignment_completion_percentage` | Percentage of assignments completed | 45–100 |

### Target variable

The target variable is `performance_category`, which contains three classes:

- `Low`
- `Medium`
- `High`

The labels are generated from a weighted composite score based on the four input features, with controlled Gaussian noise.

The current training script uses the following weights:

- Previous exam score: 35%
- Attendance percentage: 25%
- Assignment completion: 25%
- Study hours: 15%

The resulting noisy composite score is categorized as follows:

| Composite score | Performance category |
|---|---|
| Below 60 | Low |
| 60 to below 78 | Medium |
| 78 and above | High |

These rules define the synthetic labels used for this demonstration. They are not validated educational standards.

### Dataset distribution

The generated dataset contained the following class distribution during the recorded training run:

| Category | Records | Percentage |
|---|---:|---:|
| Low | 93 | 18.6% |
| Medium | 300 | 60.0% |
| High | 107 | 21.4% |
| **Total** | **500** | **100%** |

The dataset is saved automatically to `data/student_performance.csv` when the training script runs.

## 📁 Project Structure

```text
FAI-Student-Performance-Prediction/
│
├── app.py
│   └── Streamlit dashboard and prediction interface
│
├── train_model.py
│   └── Synthetic data generation, model training, and evaluation
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── .gitignore
│   └── Excludes temporary files and generated model binaries
│
├── data/
│   ├── .gitkeep
│   └── student_performance.csv
│
├── models/
│   ├── .gitkeep
│   └── model_metadata.json
│
└── screenshots/
    ├── dashboard.png
    └── prediction-result.png
```

**Note:** The `screenshots/` folder is a suggested location for the application screenshots. Create it and add your images to use the Markdown references below. The trained `student_performance_model.pkl` file is generated locally and excluded from Git.

## 📈 Model Evaluation

The model was evaluated on a held-out test set containing 100 samples, representing 20% of the generated dataset.

### Test accuracy

| Evaluation metric | Recorded result |
|---|---:|
| Test accuracy | **76.00%** |
| Training samples | 400 |
| Test samples | 100 |

Accuracy is the proportion of test samples classified correctly.

### Classification report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Low | 0.75 | 0.47 | 0.58 | 19 |
| Medium | 0.74 | 0.93 | 0.82 | 60 |
| High | 0.92 | 0.52 | 0.67 | 21 |
| **Macro average** | **0.80** | **0.64** | **0.69** | **100** |
| **Weighted average** | **0.78** | **0.76** | **0.74** | **100** |

### Confusion matrix

Rows represent actual classes and columns represent predicted classes.

| Actual / Predicted | Low | Medium | High |
|---|---:|---:|---:|
| Low | 9 | 10 | 0 |
| Medium | 3 | 56 | 1 |
| High | 0 | 10 | 11 |

### Interpretation

- The model correctly classified 76 out of 100 test samples.
- The Medium category achieved the highest recall at 0.93.
- The model correctly identified 9 of the 19 actual Low-category samples.
- The model correctly identified 11 of the 21 actual High-category samples.

These results show that performance differs across classes. In particular, some Low and High samples were classified as Medium, so accuracy alone is insufficient to assess the model.

**Important:** These metrics describe performance on the synthetic dataset generated by this project. They do not establish accuracy on real student data.


## 🚀 Installation and Setup

Follow these instructions to download, install, train, and run the project on your computer.

### Prerequisites

Install the following:

- Python 3.9 or a compatible newer version
- Git (optional, if downloading the ZIP archive)
- A web browser

### Option 1: Download the project as a ZIP

1. Open the GitHub repository:

   https://github.com/Mlavankumar/FAI-Student-Performance-Prediction

2. Click the green **Code** button.
3. Select **Download ZIP**.
4. Extract the downloaded archive.
5. Open a terminal in the extracted `FAI-Student-Performance-Prediction` folder.

### Option 2: Clone the repository using Git

Open a terminal and run:

```bash
git clone https://github.com/Mlavankumar/FAI-Student-Performance-Prediction.git
cd FAI-Student-Performance-Prediction
```

### Step 1: Create a virtual environment

**Windows — PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell prevents activation, you can use Command Prompt instead:

```bat
venv\Scripts\activate.bat
```

**macOS/Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2: Install dependencies

Run:

```bash
python -m pip install -r requirements.txt
```

Wait until all required packages have been installed.

### Step 3: Train the model

Run:

```bash
python train_model.py
```

The script generates the synthetic dataset, splits it into training and testing sets, trains the Decision Tree, evaluates its performance, and saves the artifacts.

Expected generated files:

```text
data/student_performance.csv
models/student_performance_model.pkl
models/model_metadata.json
```

The test accuracy and classification report will be printed in the terminal.

**Why is model training required?** The trained `.pkl` file is intentionally excluded from GitHub. Running the training script creates this file on your computer.

### Step 4: Launch the Streamlit application

Run:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal, typically:

```text
http://localhost:8501
```

Open that URL in your browser if it does not open automatically.

### Step 5: Stop the application

To stop the local Streamlit server, press `Ctrl + C` in the terminal.

## 🧪 How to Use the Application

1. Launch the application with `streamlit run app.py`.
2. Open the local URL displayed by Streamlit.
3. Navigate to the **Predict Performance** section.
4. Adjust the four input sliders:
   - Study hours per day
   - Attendance percentage
   - Previous exam score
   - Assignment completion percentage
5. Click **Predict Performance**.
6. View the predicted performance category.
7. Review the confidence estimates for Low, Medium, and High.
8. Read the general educational guidance associated with the predicted category.
9. Open the **AI & Feature Explanation** tab to review the input features and Decision Tree concepts.

You can change the inputs and make additional predictions without retraining the model.

## 📝 Sample Output

The following is an illustrative example from the recorded application demonstration.

### Sample inputs

| Input | Value |
|---|---:|
| Study hours per day | 3.0 |
| Attendance percentage | 90% |
| Previous exam score | 80 |
| Assignment completion | 75% |

### Recorded prediction

| Output | Value |
|---|---|
| Predicted category | Medium Performance |
| Low confidence | 13.3% |
| Medium confidence | 80.8% |
| High confidence | 5.8% |

The application recommended increasing study time, improving assignment completion, and reviewing previous examination mistakes.

The output above represents one specific input combination. Predictions and confidence estimates can change when the inputs change.

## ⚠️ Limitations

1. **Synthetic dataset:** The model uses generated data rather than verified student records.
2. **Limited features:** Only four indicators are included; other relevant factors are not modeled.
3. **Class imbalance:** The generated dataset contains more Medium samples than Low or High samples.
4. **Uneven class performance:** The recorded test results show lower recall for Low and High categories than for Medium.
5. **Limited generalization:** The reported test accuracy applies only to the synthetic test set.
6. **No causal inference:** The model does not prove that changing a particular input will cause a student’s performance to improve.
7. **Probability calibration:** The displayed class probabilities are model estimates and are not guaranteed to represent calibrated real-world probabilities.

## 🔮 Future Enhancements

- Evaluate the approach using a suitable, documented real-world educational dataset.
- Compare Decision Trees with Random Forest, Logistic Regression, and other classification algorithms.
- Add confusion matrix and feature-importance visualizations to the dashboard.
- Improve performance for underrepresented classes.
- Add cross-validation and hyperparameter tuning.
- Include additional features when supported by an appropriate dataset.
- Add downloadable prediction reports.
- Improve accessibility and responsive dashboard design.
- Deploy the application to a suitable hosting platform for remote demonstrations.

## 🏁 Conclusion

This project demonstrates the implementation of a supervised machine learning pipeline for classifying student performance using a Decision Tree Classifier.

The project covers synthetic dataset generation, feature-based classification, train-test splitting, model evaluation, model persistence, and interactive prediction through Streamlit.

The recorded model achieved **76.00% test accuracy** on a held-out subset of the generated demonstration dataset. The interactive dashboard makes the prediction process accessible and provides an example of how machine learning models can be integrated into simple applications.

The project serves as an educational demonstration of machine learning concepts within the Fundamentals of Artificial Intelligence case study.

## ⚖️ Academic Disclaimer

This application is intended strictly for educational and demonstration purposes. The model was trained on synthetic data created by this project and has not been validated for real-world academic use.

Predictions must not be used to label students, make high-stakes educational decisions, or replace professional judgment. Real-world use would require appropriate data, validation, privacy safeguards, fairness evaluation, and domain expert review.

## 👨‍💻 Author and Repository

**Project:** Student Performance Prediction Using a Decision Tree Classifier  
**Course:** Fundamentals of Artificial Intelligence (FAI)  
**Repository:** [FAI-Student-Performance-Prediction](https://github.com/Mlavankumar/FAI-Student-Performance-Prediction)

Source code, training instructions, dataset generation, and application setup are available in this repository.

---

<p align="center">
  Developed as an academic case study to demonstrate supervised machine learning and interactive AI application development.
</p>

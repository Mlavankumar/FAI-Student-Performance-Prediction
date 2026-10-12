# Student Performance Prediction Using a Decision Tree Classifier

## Fundamentals of Artificial Intelligence (FAI) Case Study

### Objective
The objective of this project is to build an interpretable machine learning model that predicts student academic performance based on key academic and behavioral indicators such as study time, past failures, attendance/absences, and prior scores. By utilizing an intuitive decision tree classification approach, educators and academic advisors can identify students needing support early and implement timely interventions.

### Machine Learning Algorithm
- **Decision Tree Classifier (`scikit-learn`)**:
  - Selected for its high transparency, interpretable decision rules, and visualizable tree hierarchy.
  - Well-suited for educational classification tasks where stakeholders require clear explanations for predictions.

### Technology Stack
- **Python**: Core programming language
- **scikit-learn**: Machine learning model training, evaluation, and metrics
- **pandas**: Data loading, cleaning, and feature engineering
- **matplotlib**: Model visualization and exploratory analysis plots
- **Streamlit**: Interactive web dashboard for user inputs and real-time prediction

### Project Structure
```text
FAI-Student-Performance-Prediction/
├── app.py              # Streamlit web application for interactive prediction
├── train_model.py      # Pipeline script for data preprocessing, training, and evaluation
├── requirements.txt    # Project dependencies
├── README.md           # Project documentation and instructions
├── .gitignore          # Git ignore specifications
├── data/               # Dataset directory
└── models/             # Directory for saved trained model artifacts
```

### Planned Execution Workflow

1. **Environment Setup**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

2. **Model Training & Generation**:
   > *Note: The trained model binary (`models/student_performance_model.pkl`) is excluded from git to avoid committing binary artifacts. Run the command below to generate the dataset and train/save the model artifact:*
   ```bash
   python train_model.py
   ```

3. **Run Streamlit Web Application**:
   ```bash
   streamlit run app.py
   ```


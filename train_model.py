"""
train_model.py
Fundamentals of Artificial Intelligence (FAI) Case Study:
Student Performance Prediction Using a Decision Tree Classifier

This script:
1. Generates a reproducible synthetic dataset of 500 student academic records.
   (Documented explicitly as synthetic demonstration data, not real student data).
2. Saves the dataset to data/student_performance.csv.
3. Splits data into training (80%) and testing (20%) sets.
4. Trains a scikit-learn DecisionTreeClassifier.
5. Evaluates model performance: prints test accuracy, classification report,
   and confusion matrix.
6. Serializes the trained model to models/student_performance_model.pkl using joblib.
7. Saves feature names and class labels to models/model_metadata.json.
"""

import json
import os
import sys

try:
    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
except ImportError as err:
    print("\n" + "!" * 70)
    print(f" [ERROR] Missing required Python package: {err.name if hasattr(err, 'name') else err}")
    print(" Please install project dependencies first using:")
    print("     pip install -r requirements.txt")
    print("!" * 70 + "\n")
    sys.exit(1)


# Constants and Configuration
RANDOM_SEED = 42
NUM_SAMPLES = 500
TEST_SIZE = 0.20
DATA_DIR = "data"
MODELS_DIR = "models"
DATA_FILEPATH = os.path.join(DATA_DIR, "student_performance.csv")
MODEL_FILEPATH = os.path.join(MODELS_DIR, "student_performance_model.pkl")
METADATA_FILEPATH = os.path.join(MODELS_DIR, "model_metadata.json")

FEATURE_NAMES = [
    "study_hours",
    "attendance_percentage",
    "previous_exam_score",
    "assignment_completion_percentage",
]
CLASS_LABELS = ["Low", "Medium", "High"]


def generate_synthetic_student_data(n_samples: int = NUM_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a reproducible synthetic dataset of student academic records.

    NOTE: This is synthetic demonstration data generated for the Fundamentals
    of Artificial Intelligence (FAI) case study and does NOT represent real student data.
    """
    rng = np.random.default_rng(seed)

    # 1. Generate realistic academic features
    # Study hours per day: between 1.0 and 9.5 hours
    study_hours = np.round(rng.uniform(1.0, 9.5, size=n_samples), 1)

    # Attendance percentage: between 50.0% and 100.0%
    attendance_pct = np.round(rng.uniform(50.0, 100.0, size=n_samples), 1)

    # Previous exam score: between 40.0 and 98.0 marks out of 100
    previous_exam = np.round(rng.uniform(40.0, 98.0, size=n_samples), 1)

    # Assignment completion percentage: between 45.0% and 100.0%
    assignment_pct = np.round(rng.uniform(45.0, 100.0, size=n_samples), 1)

    # 2. Derive performance category using intuitive weighted scoring
    # Weights: Previous Exam (35%), Attendance (25%), Assignment Completion (25%), Study Time (15%)
    normalized_study = (study_hours / 10.0) * 100.0
    composite_score = (
        0.35 * previous_exam
        + 0.25 * attendance_pct
        + 0.25 * assignment_pct
        + 0.15 * normalized_study
    )

    # Add controlled Gaussian noise to prevent purely deterministic labels
    noise = rng.normal(loc=0.0, scale=4.0, size=n_samples)
    final_score = np.clip(composite_score + noise, 0.0, 100.0)

    # 3. Categorize into Low, Medium, and High performance
    categories = []
    for score in final_score:
        if score < 60.0:
            categories.append("Low")
        elif score < 78.0:
            categories.append("Medium")
        else:
            categories.append("High")

    df = pd.DataFrame({
        "study_hours": study_hours,
        "attendance_percentage": attendance_pct,
        "previous_exam_score": previous_exam,
        "assignment_completion_percentage": assignment_pct,
        "performance_category": categories,
    })

    return df


def ensure_directories_exist() -> None:
    """Ensure data/ and models/ directories exist in the project."""
    for directory in [DATA_DIR, MODELS_DIR]:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"[OK] Created directory: {directory}/")
            except OSError as err:
                print(f"[ERROR] Failed to create directory '{directory}': {err}")
                sys.exit(1)


def main():
    print("=" * 70)
    print(" Student Performance Prediction - Decision Tree Training Pipeline")
    print(" Fundamentals of Artificial Intelligence (FAI) Case Study")
    print("=" * 70)

    # Step 1: Ensure directory structure
    ensure_directories_exist()

    # Step 2: Generate reproducible synthetic dataset
    print(f"\n[1/6] Generating synthetic dataset ({NUM_SAMPLES} samples, seed={RANDOM_SEED})...")
    print("      Note: This is synthetic demonstration data, not real student data.")
    try:
        data = generate_synthetic_student_data(n_samples=NUM_SAMPLES, seed=RANDOM_SEED)
        print("      Class distribution in generated dataset:")
        for label, count in data["performance_category"].value_counts().items():
            print(f"        - {label}: {count} students ({count / len(data) * 100:.1f}%)")
    except Exception as err:
        print(f"[ERROR] Failed to generate synthetic data: {err}")
        sys.exit(1)

    # Step 3: Save dataset to data/student_performance.csv
    print(f"\n[2/6] Saving dataset to '{DATA_FILEPATH}'...")
    try:
        data.to_csv(DATA_FILEPATH, index=False)
        print(f"      [OK] Dataset successfully saved to '{DATA_FILEPATH}'.")
    except Exception as err:
        print(f"[ERROR] Failed to save dataset to '{DATA_FILEPATH}': {err}")
        sys.exit(1)

    # Step 4: Split data into training and test sets (80:20 split)
    print(f"\n[3/6] Splitting data into Train ({int((1-TEST_SIZE)*100)}%) and Test ({int(TEST_SIZE*100)}%)...")
    X = data[FEATURE_NAMES]
    y = data["performance_category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    print(f"      Training samples: {len(X_train)}")
    print(f"      Testing samples:  {len(X_test)}")

    # Step 5: Train Decision Tree Classifier
    print("\n[4/6] Training DecisionTreeClassifier...")
    try:
        clf = DecisionTreeClassifier(
            criterion="gini",
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=RANDOM_SEED,
        )
        clf.fit(X_train, y_train)
        print("      [OK] Model training completed.")
    except Exception as err:
        print(f"[ERROR] Model training failed: {err}")
        sys.exit(1)

    # Step 6: Evaluate Model Performance
    print("\n[5/6] Evaluating Model Performance on Test Set...")
    try:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print("-" * 50)
        print(f"  Test Accuracy: {accuracy * 100:.2f}%")
        print("-" * 50)

        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, labels=CLASS_LABELS, zero_division=0))

        print("  Confusion Matrix (Rows: Actual, Columns: Predicted):")
        cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
        col_header = "       " + " ".join([f"{label:>8}" for label in CLASS_LABELS])
        print(col_header)
        for idx, row_label in enumerate(CLASS_LABELS):
            row_str = f"{row_label:>6} " + " ".join([f"{val:>8}" for val in cm[idx]])
            print(row_str)
        print("-" * 50)
    except Exception as err:
        print(f"[ERROR] Model evaluation failed: {err}")
        sys.exit(1)

    # Step 7: Serialize Trained Model and Metadata
    print("\n[6/6] Saving trained model and metadata...")
    try:
        # Save model
        joblib.dump(clf, MODEL_FILEPATH)
        print(f"      [OK] Trained model saved to '{MODEL_FILEPATH}'")

        # Save metadata
        metadata = {
            "model_type": "DecisionTreeClassifier",
            "feature_names": FEATURE_NAMES,
            "class_labels": CLASS_LABELS,
            "hyperparameters": {
                "criterion": "gini",
                "max_depth": 4,
                "min_samples_split": 5,
                "min_samples_leaf": 3,
                "random_state": RANDOM_SEED,
            },
            "test_accuracy": round(float(accuracy), 4),
            "dataset_info": {
                "file_path": DATA_FILEPATH,
                "total_samples": NUM_SAMPLES,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "disclaimer": "Synthetic demonstration data generated for academic case study purposes.",
            },
        }

        with open(METADATA_FILEPATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        print(f"      [OK] Model metadata saved to '{METADATA_FILEPATH}'")
    except Exception as err:
        print(f"[ERROR] Failed to save model artifacts: {err}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(" Training pipeline completed successfully!")
    print(f" Model ready at:    {MODEL_FILEPATH}")
    print(f" Metadata ready at: {METADATA_FILEPATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()

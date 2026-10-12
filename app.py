"""
app.py
Student Performance Prediction System
Fundamentals of Artificial Intelligence (FAI) Case Study | Decision Tree Classifier

Interactive Streamlit Application:
- Loads the trained Decision Tree Classifier from models/student_performance_model.pkl
- Loads class labels and feature metadata from models/model_metadata.json
- Handles missing model files gracefully without crashing
- Collects 4 student parameters via sliders
- Predicts performance category (Low, Medium, High) with recommendations
- Explains the Decision Tree methodology and feature relevance
- Provides clear disclaimers regarding synthetic data
"""

import json
import os
import joblib
import pandas as pd
import streamlit as st

# File paths for model artifacts
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "student_performance_model.pkl")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")


@st.cache_resource
def load_model_and_metadata():
    """
    Load the serialized model and metadata.
    Returns (model, metadata) if both exist, otherwise (None, None).
    """
    if not os.path.exists(MODEL_PATH) or not os.path.exists(METADATA_PATH):
        return None, None

    try:
        model = joblib.load(MODEL_PATH)
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as err:
        st.error(f"Error loading model artifacts: {err}")
        return None, None


def main():
    # Page configuration
    st.set_page_config(
        page_title="Student Performance Prediction System",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Clean CSS styling for dashboard cards and badges
    st.markdown(
        """
        <style>
        .sub-header {
            font-size: 1.1rem;
            color: #4b5563;
            margin-bottom: 1.5rem;
            font-weight: 500;
        }
        .result-card {
            border-radius: 12px;
            padding: 22px;
            margin-top: 16px;
            margin-bottom: 16px;
        }
        .result-high {
            background-color: #ecfdf5;
            border: 2px solid #10b981;
            color: #065f46;
        }
        .result-medium {
            background-color: #fffbeb;
            border: 2px solid #f59e0b;
            color: #92400e;
        }
        .result-low {
            background-color: #fef2f2;
            border: 2px solid #ef4444;
            color: #991b1b;
        }
        .recommendation-box {
            font-size: 1.0rem;
            line-height: 1.5;
            margin-top: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Requirement 4: Title and Subtitle
    st.title("🎓 Student Performance Prediction System")
    st.markdown(
        "<div class='sub-header'>FAI Case Study | Decision Tree Classifier</div>",
        unsafe_allow_html=True,
    )

    # Requirements 1 & 2: Load model and metadata
    model, metadata = load_model_and_metadata()

    # Requirement 3: If model files are missing, display clear instructions and do not crash
    if model is None or metadata is None:
        st.warning("⚠️ Trained Model Files Missing")
        st.info(
            "The model artifacts (`models/student_performance_model.pkl` and "
            "`models/model_metadata.json`) could not be found.\n\n"
            "Please train and serialize the model first by running:\n\n"
            "```bash\n"
            "python train_model.py\n"
            "```\n\n"
            "Once training completes successfully, refresh this page to begin predicting."
        )
        # Requirement 10: Disclaimer
        st.markdown("---")
        st.caption(
            "📌 Note: This application is part of the Fundamentals of Artificial Intelligence (FAI) case study "
            "and uses synthetic demonstration data."
        )
        return

    # Extract feature list and metadata parameters
    feature_names = metadata.get(
        "feature_names",
        [
            "study_hours",
            "attendance_percentage",
            "previous_exam_score",
            "assignment_completion_percentage",
        ],
    )
    test_accuracy = metadata.get("test_accuracy")

    # Sidebar: Model Details & Case Study Info
    with st.sidebar:
        st.header("⚙️ Model Architecture")
        st.markdown(f"**Algorithm:** `{metadata.get('model_type', 'DecisionTreeClassifier')}`")
        if test_accuracy is not None:
            st.metric("Test Accuracy", f"{test_accuracy * 100:.1f}%")

        st.markdown("---")
        st.subheader("📌 Features Considered")
        st.markdown(
            """
            1. **Study Hours / Day** (1.0 – 9.5)
            2. **Attendance** (50% – 100%)
            3. **Previous Exam Score** (40 – 98)
            4. **Assignment Completion** (45% – 100%)
            """
        )
        st.markdown("---")
        st.caption("Fundamentals of Artificial Intelligence (FAI)")

    # Main Tabs: Interactive Prediction & AI Methodology Explanation
    tab_predict, tab_explain = st.tabs(["🔮 Predict Performance", "📖 AI & Feature Explanation"])

    with tab_predict:
        st.subheader("1. Student Input Parameters")
        st.write("Adjust the sliders below to simulate a student's academic and behavioral indicators:")

        # Requirement 5: Streamlit sliders in columns
        col1, col2 = st.columns(2)

        with col1:
            study_hours = st.slider(
                "⏱️ Study Hours Per Day",
                min_value=1.0,
                max_value=9.5,
                value=5.0,
                step=0.1,
                help="Daily hours devoted to self-study and revision (1.0 to 9.5).",
            )

            attendance_pct = st.slider(
                "📅 Attendance Percentage (%)",
                min_value=50.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
                help="Class attendance rate (50% to 100%).",
            )

        with col2:
            previous_exam_score = st.slider(
                "📝 Previous Exam Score (out of 100)",
                min_value=40.0,
                max_value=98.0,
                value=70.0,
                step=1.0,
                help="Marks scored in the prior term examination (40 to 98).",
            )

            assignment_completion_pct = st.slider(
                "📋 Assignment Completion Percentage (%)",
                min_value=45.0,
                max_value=100.0,
                value=85.0,
                step=1.0,
                help="Percentage of completed coursework assignments (45% to 100%).",
            )

        # Requirement 11: Metric cards for quick review
        st.markdown("#### Input Snapshot")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Study Time", f"{study_hours:.1f} hrs/day")
        m2.metric("Attendance", f"{attendance_pct:.0f}%")
        m3.metric("Prev. Score", f"{previous_exam_score:.0f}/100")
        m4.metric("Assignments", f"{assignment_completion_pct:.0f}%")

        st.markdown("<br>", unsafe_allow_html=True)

        # Requirement 6: Prominent Predict Performance button
        predict_clicked = st.button("🔮 Predict Performance", type="primary", use_container_width=True)

        # Requirement 7: Predict Low, Medium, or High when clicked
        if predict_clicked:
            input_dict = {
                "study_hours": [study_hours],
                "attendance_percentage": [attendance_pct],
                "previous_exam_score": [previous_exam_score],
                "assignment_completion_percentage": [assignment_completion_pct],
            }
            # Maintain exact training feature order
            input_df = pd.DataFrame(input_dict)[feature_names]

            try:
                prediction = model.predict(input_df)[0]

                # Probabilities if available
                prob_dict = {}
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(input_df)[0]
                    prob_dict = {cls: prob for cls, prob in zip(model.classes_, probs)}

                # Requirement 8: Result card and appropriate recommendations
                st.markdown("### 2. Prediction Outcome")

                if prediction == "High":
                    card_class = "result-high"
                    icon = "⭐"
                    badge_title = "High Performance"
                    rec_text = (
                        "The student demonstrates strong study habits, high attendance, and consistent "
                        "assignment quality. **Recommendation:** Maintain current routine, consider advanced "
                        "study projects, and mentor peers."
                    )
                elif prediction == "Medium":
                    card_class = "result-medium"
                    icon = "⚡"
                    badge_title = "Medium Performance"
                    rec_text = (
                        "The student exhibits solid baseline capability with potential for higher achievement. "
                        "**Recommendation:** Increase daily study time by 1–2 hours, target 90%+ assignment "
                        "completion, and review past exam mistakes to reach the High tier."
                    )
                else:  # Low
                    card_class = "result-low"
                    icon = "⚠️"
                    badge_title = "Low Performance (At-Risk)"
                    rec_text = (
                        "The indicators suggest early academic risk. **Recommendation:** Immediate intervention "
                        "recommended. Prioritize class attendance, attend remedial tutoring, and focus on completing "
                        "all pending assignments."
                    )

                st.markdown(
                    f"""
                    <div class='result-card {card_class}'>
                        <h2 style='margin: 0; font-size: 1.5rem;'>{icon} Predicted Performance: {badge_title}</h2>
                        <div class='recommendation-box'>
                            <strong>Actionable Academic Advice:</strong><br>{rec_text}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Class confidence metrics
                if prob_dict:
                    st.markdown("##### Prediction Confidence Across Classes")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Low Confidence", f"{prob_dict.get('Low', 0.0) * 100:.1f}%")
                    c2.metric("Medium Confidence", f"{prob_dict.get('Medium', 0.0) * 100:.1f}%")
                    c3.metric("High Confidence", f"{prob_dict.get('High', 0.0) * 100:.1f}%")

            except Exception as err:
                st.error(f"Prediction failed: {err}")

    # Requirement 9: Explanation of features and Decision Tree decision rules
    with tab_explain:
        st.subheader("Decision Tree Classification Mechanics")
        st.write(
            "In this Fundamentals of Artificial Intelligence (FAI) case study, "
            "a **Decision Tree Classifier** maps non-linear educational factors into clear, "
            "interpretable decision rules."
        )

        col_feat, col_tree = st.columns(2)

        with col_feat:
            st.markdown("#### 📌 Input Features Explained")
            st.markdown(
                """
                - **Study Hours Per Day**: Measures daily independent learning and cognitive reinforcement.
                - **Attendance Percentage**: Acts as a proxy for lecture comprehension, engagement, and discipline.
                - **Previous Exam Score**: Captures cumulative baseline subject mastery and prior performance.
                - **Assignment Completion Percentage**: Reflects continuous assessment diligence and practice consistency.
                """
            )

        with col_tree:
            st.markdown("#### 🌳 How Decision Trees Make Decisions")
            st.markdown(
                """
                - **Hierarchical Splitting**: The tree evaluates features sequentially (e.g., *Is Previous Score < 65?*).
                - **Gini Impurity Minimization**: Splits are selected to maximize separation between Low, Medium, and High groups.
                - **Transparent Paths**: Every output corresponds to a readable chain of boolean conditions from root to leaf.
                - **Interpretability**: Unlike black-box models, educators can trace exactly why a student was classified as at-risk.
                """
            )

    # Requirement 10: Synthetic data disclaimer
    st.markdown("---")
    st.info(
        "📌 **Academic Disclaimer**: This model was trained on synthetic demonstration data created "
        "for the Fundamentals of Artificial Intelligence (FAI) case study. It is intended solely "
        "for educational and illustrative purposes and must not be used to make real academic decisions."
    )


if __name__ == "__main__":
    main()

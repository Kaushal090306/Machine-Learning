from streamlit_project_core import run_project_app


run_project_app(
    project_name="Diabetes Prediction",
    target_candidates=["Outcome", "outcome", "Diabetes", "diabetes"],
    project_note="Upload diabetes data to explore feature relationships and predict diabetes outcome.",
)

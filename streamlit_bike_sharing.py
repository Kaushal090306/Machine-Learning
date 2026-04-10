from streamlit_project_core import run_project_app


run_project_app(
    project_name="Bike Sharing Analysis",
    target_candidates=["cnt", "count", "registered", "casual"],
    project_note="Upload bike sharing data to analyze demand patterns and predict rentals.",
)

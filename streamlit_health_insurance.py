from streamlit_project_core import run_project_app


run_project_app(
    project_name="Health Insurance Analysis",
    target_candidates=["charges", "Charges", "claim_amount"],
    project_note="Upload insurance data to analyze costs and predict medical charges.",
)

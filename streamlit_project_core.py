import math
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _infer_task_type(y: pd.Series, requested_task: str) -> str:
    if requested_task in {"Classification", "Regression"}:
        return requested_task

    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_object_dtype(y):
        return "Classification"

    unique_count = y.nunique(dropna=True)
    if unique_count <= 20 and pd.api.types.is_integer_dtype(y):
        return "Classification"

    return "Regression"


def _build_pipeline(X: pd.DataFrame, task_type: str) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )

    if task_type == "Classification":
        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)

    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _select_default_target(columns: List[str], target_candidates: List[str]) -> str:
    for candidate in target_candidates:
        if candidate in columns:
            return candidate
    return columns[-1]


def _train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, float]]:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if X.empty:
        raise ValueError("No feature columns available after selecting the target.")

    stratify = y if task_type == "Classification" and y.nunique(dropna=True) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify,
    )

    pipeline = _build_pipeline(X_train, task_type)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics: Dict[str, float] = {}

    if task_type == "Classification":
        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["f1_weighted"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        unique_test = pd.Series(y_test).nunique(dropna=True)
        if unique_test == 2:
            try:
                y_proba = pipeline.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except Exception:
                pass
    else:
        metrics["r2"] = float(r2_score(y_test, y_pred))
        metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        metrics["rmse"] = float(rmse)

    return pipeline, X_train, X_test, y_train, y_test, metrics


def _plot_target_distribution(df: pd.DataFrame, target_col: str, task_type: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    if task_type == "Classification":
        counts = df[target_col].value_counts(dropna=False)
        sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette="Blues_d")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Count")
    else:
        sns.histplot(df[target_col], kde=True, ax=ax, color="#1f77b4")
        ax.set_xlabel(target_col)
        ax.set_ylabel("Frequency")
    ax.set_title(f"Target Distribution: {target_col}")
    st.pyplot(fig)


def _plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        st.info("Not enough numeric columns for a correlation heatmap.")
        return

    cols = numeric_df.columns.tolist()[:20]
    corr = numeric_df[cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap (first 20 numeric columns)")
    st.pyplot(fig)


def _plot_feature_relationship(df: pd.DataFrame, target_col: str, task_type: str) -> None:
    feature_cols = [col for col in df.columns if col != target_col]
    numeric_features = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    if not numeric_features:
        st.info("No numeric feature available for relationship plot.")
        return

    first_feature = numeric_features[0]
    fig, ax = plt.subplots(figsize=(8, 4))

    if task_type == "Classification":
        sns.boxplot(data=df, x=target_col, y=first_feature, ax=ax, palette="Set2")
        ax.set_xlabel(target_col)
        ax.set_ylabel(first_feature)
    else:
        sns.scatterplot(data=df, x=first_feature, y=target_col, ax=ax, alpha=0.65)
        sns.regplot(data=df, x=first_feature, y=target_col, ax=ax, scatter=False, color="red")

    ax.set_title(f"{first_feature} vs {target_col}")
    st.pyplot(fig)


def _plot_confusion_matrix(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred = model.predict(X_test)
    labels = sorted(pd.Series(y_test).dropna().unique().tolist())
    if len(labels) < 2:
        st.info("Confusion matrix requires at least two classes in test data.")
        return

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def _render_prediction_form(model: Pipeline, X_reference: pd.DataFrame, task_type: str) -> None:
    st.subheader("Prediction Input")
    st.write("Fill in feature values and click Predict.")

    with st.form("prediction_form"):
        input_data = {}
        for col in X_reference.columns:
            series = X_reference[col]
            if pd.api.types.is_numeric_dtype(series):
                default_val = float(series.median()) if not series.dropna().empty else 0.0
                min_val = float(series.min()) if not series.dropna().empty else None
                max_val = float(series.max()) if not series.dropna().empty else None
                step_val = 1.0 if pd.api.types.is_integer_dtype(series) else 0.01

                input_data[col] = st.number_input(
                    label=col,
                    value=default_val,
                    min_value=min_val,
                    max_value=max_val,
                    step=step_val,
                )
            else:
                uniques = series.dropna().astype(str).unique().tolist()
                if not uniques:
                    uniques = ["Unknown"]
                input_data[col] = st.selectbox(label=col, options=sorted(uniques)[:100])

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    st.write(f"Predicted value: {prediction}")

    if task_type == "Classification":
        try:
            probabilities = model.predict_proba(input_df)[0]
            classes = model.named_steps["model"].classes_
            prob_df = pd.DataFrame({"class": classes, "probability": probabilities}).sort_values(
                by="probability", ascending=False
            )
            st.dataframe(prob_df, use_container_width=True)
        except Exception:
            pass


def run_project_app(project_name: str, target_candidates: Optional[List[str]] = None, project_note: str = "") -> None:
    target_candidates = target_candidates or []

    st.set_page_config(page_title=project_name, layout="wide")
    st.title(project_name)
    if project_note:
        st.write(project_note)

    st.sidebar.header("Data")
    st.sidebar.write("Upload a CSV file for this project.")
    uploaded_file = st.sidebar.file_uploader("Upload dataset", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file from the sidebar to begin analysis and prediction.")
        st.stop()

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV file: {exc}")
        st.stop()

    if df.empty:
        st.error("Uploaded dataset is empty.")
        st.stop()

    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
    with col2:
        st.write("Data types")
        st.dataframe(df.dtypes.astype(str).rename("dtype"), use_container_width=True)

    st.dataframe(df.head(10), use_container_width=True)

    st.sidebar.header("Model Setup")
    default_target = _select_default_target(df.columns.tolist(), target_candidates)
    target_col = st.sidebar.selectbox("Target column", options=df.columns.tolist(), index=df.columns.tolist().index(default_target))

    task_request = st.sidebar.selectbox("Task type", options=["Auto", "Classification", "Regression"], index=0)
    task_type = _infer_task_type(df[target_col], task_request)

    st.sidebar.write(f"Detected task: {task_type}")

    try:
        model, X_train, X_test, y_train, y_test, metrics = _train_and_evaluate(df, target_col, task_type)
    except Exception as exc:
        st.error(f"Model training failed: {exc}")
        st.stop()

    st.subheader("Model Metrics")
    if task_type == "Classification":
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metrics.get('accuracy', float('nan')):.4f}")
        c2.metric("Weighted F1", f"{metrics.get('f1_weighted', float('nan')):.4f}")
        c3.metric("ROC AUC", f"{metrics.get('roc_auc', float('nan')):.4f}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("R2", f"{metrics.get('r2', float('nan')):.4f}")
        c2.metric("MAE", f"{metrics.get('mae', float('nan')):.4f}")
        c3.metric("RMSE", f"{metrics.get('rmse', float('nan')):.4f}")

    st.subheader("Analysis Plots")
    tab1, tab2, tab3 = st.tabs(["Target Distribution", "Correlation Heatmap", "Feature Relationship"])
    with tab1:
        _plot_target_distribution(df, target_col, task_type)
    with tab2:
        _plot_correlation_heatmap(df)
    with tab3:
        _plot_feature_relationship(df, target_col, task_type)

    if task_type == "Classification":
        st.subheader("Classification Diagnostics")
        _plot_confusion_matrix(model, X_test, y_test)

    _render_prediction_form(model, X_train, task_type)

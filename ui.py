import base64
import streamlit as st
import requests
import json
import altair as alt
import pandas as pd
import io

API_ENDPOINT = st.secrets["api"]["endpoint"]


def start_job(bucket, key):
    """Start a new evaluation job."""
    try:
        response = requests.post(
            f"{API_ENDPOINT}/jobs", json={"bucket": bucket, "key": key}
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error starting job: {e}")
        return None


def upload_file(uploaded_file):
    """Upload file to S3 via API Gateway."""
    try:
        file_content = uploaded_file.getvalue()
        base64_content = base64.b64encode(file_content).decode("utf-8")
        payload = {"filename": uploaded_file.name, "file": base64_content}
        response = requests.post(f"{API_ENDPOINT}/uploads", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Upload error: {e}")
        return None


def get_job_results(job_id):
    """Fetch job results."""
    try:
        response = requests.get(f"{API_ENDPOINT}/jobs/{job_id}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching job results: {e}")
        return None


def create_jsonl_from_prompts(prompts_data):
    """Create a JSONL file from the prompt data."""
    output = io.StringIO()
    for prompt in prompts_data:
        output.write(json.dumps(prompt) + "\n")
    return output


# Main App
def main():
    st.set_page_config(
        page_title="LENS",
        page_icon=":material/circle:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("LENS: LLM Evaluation & Scoring")

    # Initialize session state for prompt creator
    if "prompts_data" not in st.session_state:
        st.session_state.prompts_data = []
    if "show_prompt_creator" not in st.session_state:
        st.session_state.show_prompt_creator = False
    if "current_prompt" not in st.session_state:
        st.session_state.current_prompt = {"id": "", "prompt": ""}
    if "editing_index" not in st.session_state:
        st.session_state.editing_index = -1

    # Sidebar: Prompt Creator Button
    st.sidebar.header("Create Prompts")
    if st.sidebar.button("Open Prompt Creator"):
        st.session_state.show_prompt_creator = True

    # Prompt Creator Modal
    if st.session_state.show_prompt_creator:
        with st.sidebar.expander("Prompt Creator", expanded=True):
            st.subheader("Create JSONL Prompts")

            # Prompt list
            if st.session_state.prompts_data:
                st.write("Current Prompts:")
                for i, prompt in enumerate(st.session_state.prompts_data):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{prompt['id']}**")
                    with col2:
                        if st.button("Edit", key=f"edit_{i}"):
                            st.session_state.current_prompt = prompt.copy()
                            st.session_state.editing_index = i
                        if st.button("Delete", key=f"delete_{i}"):
                            st.session_state.prompts_data.pop(i)
                            st.rerun()

            # Form for adding/editing prompts
            with st.form(key="prompt_form"):
                st.text_input(
                    "Prompt ID",
                    key="prompt_id",
                    value=st.session_state.current_prompt["id"],
                )
                st.text_area(
                    "Prompt",
                    key="prompt_text",
                    value=st.session_state.current_prompt["prompt"],
                )

                col1, col2 = st.columns(2)
                with col1:
                    submit_button = st.form_submit_button("Save Prompt")
                with col2:
                    cancel_button = st.form_submit_button("Cancel")

            if submit_button:
                prompt_data = {
                    "id": st.session_state.prompt_id,
                    "prompt": st.session_state.prompt_text,
                }

                if st.session_state.editing_index >= 0:
                    # Update existing prompt
                    st.session_state.prompts_data[st.session_state.editing_index] = (
                        prompt_data
                    )
                    st.session_state.editing_index = -1
                else:
                    # Add new prompt
                    st.session_state.prompts_data.append(prompt_data)

                # Reset current prompt
                st.session_state.current_prompt = {"id": "", "prompt": ""}
                st.rerun()

            if cancel_button:
                st.session_state.current_prompt = {"id": "", "prompt": ""}
                st.session_state.editing_index = -1
                st.rerun()

            # Generate and upload JSONL file
            if st.session_state.prompts_data:
                if st.button("Generate and Upload JSONL"):
                    jsonl_file = create_jsonl_from_prompts(
                        st.session_state.prompts_data
                    )

                    # Create a BytesIO object for upload
                    jsonl_bytes = io.BytesIO(jsonl_file.getvalue().encode())
                    jsonl_bytes.name = "prompts.jsonl"

                    # Upload the file
                    upload_response = upload_file(jsonl_bytes)
                    if (
                        upload_response
                        and "bucket" in upload_response
                        and "key" in upload_response
                    ):
                        st.success(
                            f"File uploaded. File ID: {upload_response['filename']}"
                        )
                        st.session_state["uploaded_bucket"] = upload_response["bucket"]
                        st.session_state["uploaded_key"] = upload_response["key"]
                        st.session_state.show_prompt_creator = False
                        st.rerun()

                # Download option
                jsonl_file = create_jsonl_from_prompts(st.session_state.prompts_data)
                st.download_button(
                    label="Download JSONL File",
                    data=jsonl_file.getvalue(),
                    file_name="prompts.jsonl",
                    mime="application/json",
                )

            # Close button
            if st.button("Close Prompt Creator"):
                st.session_state.show_prompt_creator = False
                st.rerun()

    # Sidebar: Upload Prompts File
    st.sidebar.header("Upload Prompts File")
    uploaded_file = st.sidebar.file_uploader("Choose a JSONL file", type=["jsonl"])
    if uploaded_file is not None:
        if st.sidebar.button("Upload and Evaluate"):
            upload_response = upload_file(uploaded_file)
            if (
                upload_response
                and "bucket" in upload_response
                and "key" in upload_response
            ):
                st.sidebar.success(
                    f"File uploaded. File ID: {upload_response['filename']}"
                )
                st.session_state["uploaded_bucket"] = upload_response["bucket"]
                st.session_state["uploaded_key"] = upload_response["key"]

    # Sidebar: Start New Evaluation Job
    st.sidebar.header("Start New Evaluation Job")
    bucket = st.sidebar.text_input(
        "S3 Bucket", value=st.session_state.get("uploaded_bucket", "llm-eval-candle")
    )
    key = st.sidebar.text_input(
        "S3 Key", value=st.session_state.get("uploaded_key", "prompts.jsonl")
    )
    if st.sidebar.button("Start Evaluation Job"):
        job_response = start_job(bucket, key)
        if job_response:
            if isinstance(job_response["body"], str):
                body = json.loads(job_response["body"])
            else:
                body = job_response["body"]
            st.sidebar.success(f"Job started! Job ID: {body['job_id']}")
            st.session_state["last_job_id"] = body["job_id"]

    # Main Area: Job Results
    st.header("Job Results")
    job_id = st.text_input(
        "Enter Job ID", value=st.session_state.get("last_job_id", "")
    )

    # Fetch Results Button
    if st.button("Fetch Results"):
        if job_id:
            with st.spinner("Fetching results..."):
                results = get_job_results(job_id)
                if results:
                    st.session_state["results"] = results
                    # st.success("Results fetched successfully!")
                else:
                    st.error("Failed to fetch results.")

    # Render Results if Available
    if "results" in st.session_state:
        results = st.session_state["results"]
        status = results.get("status", "UNKNOWN")

        # Job Status
        st.subheader("Job Status")
        if status == "COMPLETED":
            st.success(f"Status: {status}")
        elif status == "PENDING":
            st.warning(f"Status: {status}")
        else:
            st.error(f"Status: {status}")

        # Raw JSON Data
        with st.expander("View Raw JSON Data"):
            st.json(results)

        # Parse Results Data
        results_data = results.get("results", "{}")
        if isinstance(results_data, str):
            results_data = json.loads(results_data)

        if results_data:
            # Create DataFrame for Metrics
            metrics_data = []
            for prompt_id, prompt_result in results_data.items():
                if isinstance(prompt_result, dict) and "metrics" in prompt_result:
                    metrics = prompt_result["metrics"]
                    metrics["prompt_id"] = prompt_id
                    metrics_data.append(metrics)
            if metrics_data:
                metrics_df = pd.DataFrame(metrics_data)

                # Metrics Comparison Table
                st.subheader("Metrics Comparison")
                styled_df = metrics_df.set_index("prompt_id").style.format("{:.3f}")
                st.dataframe(styled_df, use_container_width=True)

                # Performance Comparison Chart
                st.subheader("Performance Comparison")

                # Metric Selector
                available_metrics = ["composite", "accuracy", "rougeL", "bleu"]
                if "selected_metric" not in st.session_state:
                    st.session_state["selected_metric"] = "composite"

                selected_metric = st.selectbox(
                    "Select metric to visualize:",
                    options=available_metrics,
                    index=available_metrics.index(st.session_state["selected_metric"]),
                    key="metric_selector",
                    on_change=lambda: st.session_state.update(
                        selected_metric=st.session_state["metric_selector"]
                    ),
                    format_func=lambda x: {
                        "composite": "Composite Score",
                        "accuracy": "Accuracy",
                        "rougeL": "ROUGE-L Score",
                        "bleu": "BLEU Score",
                    }.get(x, x.title()),
                )

                # Prepare Chart Data
                chart_data = metrics_df[["prompt_id", selected_metric]].copy()
                chart_data["prompt_name"] = chart_data["prompt_id"].apply(
                    lambda x: x.replace("_", " ").title()
                )
                chart_data = chart_data.sort_values(selected_metric, ascending=False)

                # Create Bar Chart
                chart_titles = {
                    "composite": "Composite Performance Score",
                    "accuracy": "Accuracy Score",
                    "rougeL": "ROUGE-L Score",
                    "bleu": "BLEU Score",
                }
                chart_title = f"{chart_titles.get(selected_metric, selected_metric.title())} by Prompt Type"
                metric_chart = (
                    alt.Chart(chart_data)
                    .mark_bar()
                    .encode(
                        x=alt.X("prompt_name:N", title="Prompt Type", sort=None),
                        y=alt.Y(
                            f"{selected_metric}:Q",
                            title=chart_titles.get(
                                selected_metric, selected_metric.title()
                            ),
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        color=alt.Color(
                            f"{selected_metric}:Q",
                            scale=alt.Scale(scheme="viridis"),
                            legend=None,
                        ),
                        tooltip=[
                            "prompt_name",
                            alt.Tooltip(f"{selected_metric}:Q", format=".3f"),
                        ],
                    )
                    .properties(title=chart_title, height=400)
                )
                st.altair_chart(metric_chart, use_container_width=True)

                # Summary Insights
                st.subheader("Key Insights")
                best_overall = metrics_df.loc[metrics_df["composite"].idxmax()][
                    "prompt_id"
                ]
                best_accuracy = metrics_df.loc[metrics_df["accuracy"].idxmax()][
                    "prompt_id"
                ]
                best_rouge = metrics_df.loc[metrics_df["rougeL"].idxmax()]["prompt_id"]
                best_bleu = metrics_df.loc[metrics_df["bleu"].idxmax()]["prompt_id"]
                insights_col1, insights_col2 = st.columns(2)
                with insights_col1:
                    st.info(f"🏆 Best overall performance: **{best_overall}**")
                    st.info(f"⭐ Highest accuracy: **{best_accuracy}**")
                with insights_col2:
                    st.info(f"📝 Best ROUGE-L score: **{best_rouge}**")
                    st.info(f"🔤 Best BLEU score: **{best_bleu}**")

            # LangSmith Links
            st.subheader("LangSmith Evaluation Links")
            for prompt_id, prompt_result in results_data.items():
                if (
                    isinstance(prompt_result, dict)
                    and "langsmith_link" in prompt_result
                ):
                    link_text = prompt_result["langsmith_link"]
                    url = None
                    if "https://" in link_text:
                        url = link_text.split("https://")[1].split("\n")[0]
                        url = "https://" + url
                    if url:
                        st.markdown(
                            f"**{prompt_id.title()}**: [View evaluation results]({url})"
                        )

            # Detailed Metrics per Prompt
            st.subheader("Detailed Metrics")
            for prompt_id, prompt_result in results_data.items():
                with st.expander(f"Prompt: {prompt_id.replace('_', ' ').title()}"):
                    if isinstance(prompt_result, dict) and "metrics" in prompt_result:
                        metrics = prompt_result["metrics"]
                        metric_cols = st.columns(len(metrics))
                        for i, (metric_name, metric_value) in enumerate(
                            metrics.items()
                        ):
                            with metric_cols[i]:
                                try:
                                    formatted_value = f"{float(metric_value):.3f}"
                                except (ValueError, TypeError):
                                    formatted_value = str(metric_value)
                                st.metric(
                                    label=metric_name.upper(),
                                    value=formatted_value,
                                    delta_color="normal",
                                )
                        st.json(prompt_result)


if __name__ == "__main__":
    main()

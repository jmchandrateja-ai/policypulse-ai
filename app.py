import gradio as gr
from orchestrator import Orchestrator

orchestrator = Orchestrator()

def analyze_complaint(complaint_text):
    if not complaint_text.strip():
        return "Please enter a complaint.", "", "", "", ""

    state = orchestrator.run(complaint_text)

    thought_log = ""
    for entry in state.thought_log:
        thought_log += f"[{entry['time']}] {entry['agent']}: {entry['action']}\n"
        if entry['detail']:
            thought_log += f"  -> {entry['detail']}\n"

    summary = f"""Domain    : {state.domain.upper()}
Sentiment : {state.sentiment.upper()}
Stance    : {state.stance.upper()}
Urgency   : {state.urgency.upper()}
Routing   : {state.routing_decision.upper()}
Abusive   : {"Yes — cleaned" if state.is_abusive else "No"}"""

    return (
        state.moderated_text,
        summary,
        thought_log,
        state.legislator_brief
    )

with gr.Blocks(title="PolicyPulse AI", theme=gr.themes.Default()) as app:

    gr.Markdown("""
    # PolicyPulse AI
    ### Agentic Citizen Voice and Policy Intelligence Platform
    Submit a policy complaint. The system analyzes it and generates a legislator intelligence brief.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Citizen Input")
            complaint_input = gr.Textbox(
                label="Enter your policy complaint",
                placeholder="e.g. The new property tax is too high for retired citizens...",
                lines=4
            )

            gr.Examples(
                examples=[
                    ["The new property tax is too high and unaffordable for retired citizens"],
                    ["This stupid government is ruining the roads with potholes everywhere"],
                    ["The government hospital has no medicine and doctors are absent. This is urgent!"],
                    ["I support the new metro expansion. It will help daily commuters greatly."],
                    ["School fees have increased 40% this year. Poor families cannot afford education."],
                ],
                inputs=complaint_input
            )

            submit_btn = gr.Button("Analyze Complaint", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Analysis Results")
            cleaned_output = gr.Textbox(label="Cleaned Text", lines=2)
            summary_output = gr.Textbox(label="Intelligence Summary", lines=7)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Agent Thought Log")
            thought_log_output = gr.Textbox(label="Agent Reasoning", lines=15)

        with gr.Column(scale=1):
            gr.Markdown("### Legislator Brief")
            brief_output = gr.Textbox(label="Full Report", lines=15)

    submit_btn.click(
        fn=analyze_complaint,
        inputs=complaint_input,
        outputs=[cleaned_output, summary_output, thought_log_output, brief_output]
    )

if __name__ == "__main__":
    app.launch()

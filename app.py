import gradio as gr
import os
from orchestrator import Orchestrator

orchestrator = Orchestrator()

def analyze_text(complaint_text):
    if not complaint_text or not complaint_text.strip():
        return "Please enter a complaint.", "", "", "", None

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

    audio = state.confirmation_audio if state.confirmation_audio else None

    return (
        state.moderated_text,
        summary,
        thought_log,
        state.legislator_brief,
        audio
    )

def analyze_voice(audio_path):
    if audio_path is None:
        return "No audio recorded.", "", "", "", None

    import whisper
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    transcript = result["text"].strip()

    if not transcript:
        return "Could not transcribe audio.", "", "", "", None

    return analyze_text(transcript)

def read_policy(policy_text):
    if not policy_text.strip():
        return None
    try:
        from gtts import gTTS
        import time
        os.makedirs("outputs/audio", exist_ok=True)
        tts = gTTS(text=policy_text, lang="en", slow=False)
        path = f"outputs/audio/reader_{int(time.time())}.mp3"
        tts.save(path)
        return path
    except Exception as e:
        return None

with gr.Blocks(title="PolicyPulse AI") as app:

    gr.Markdown("""
    # PolicyPulse AI
    ### Agentic Citizen Voice and Policy Intelligence Platform
    """)

    with gr.Tabs():

        # TAB 1 — Citizen Complaint
        with gr.Tab("Submit Complaint"):
            gr.Markdown("### Submit Your Policy Complaint")

            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Type your complaint",
                        placeholder="e.g. The new property tax is too high...",
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
                        inputs=text_input
                    )
                    text_btn = gr.Button("Analyze Text Complaint", variant="primary")

                with gr.Column():
                    gr.Markdown("### Or Record Voice Complaint")
                    voice_input = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Record your complaint in Hindi, Kannada or English"
                    )
                    voice_btn = gr.Button("Analyze Voice Complaint", variant="secondary")

            with gr.Row():
                cleaned_out  = gr.Textbox(label="Cleaned Text", lines=2)
                summary_out  = gr.Textbox(label="Intelligence Summary", lines=7)

            with gr.Row():
                log_out      = gr.Textbox(label="Agent Thought Log", lines=12)
                brief_out    = gr.Textbox(label="Legislator Brief", lines=12)

            gr.Markdown("### Confirmation Message")
            audio_out = gr.Audio(label="Your complaint confirmation", autoplay=True)

            text_btn.click(
                fn=analyze_text,
                inputs=text_input,
                outputs=[cleaned_out, summary_out, log_out, brief_out, audio_out]
            )
            voice_btn.click(
                fn=analyze_voice,
                inputs=voice_input,
                outputs=[cleaned_out, summary_out, log_out, brief_out, audio_out]
            )

        # TAB 2 — Policy Voice Reader
        with gr.Tab("Policy Voice Reader"):
            gr.Markdown("### Paste any government policy text to hear it in plain language")
            policy_input = gr.Textbox(
                label="Paste policy text here",
                placeholder="Paste any government notification or policy document...",
                lines=6
            )
            gr.Examples(
                examples=[
                    ["The property tax shall be liable to revision pursuant to the provisions of the Municipal Act in accordance with the schedule annexed hereto."],
                    ["All citizens entitled to benefits under the scheme are required to submit applications prior to the expiry of the aforementioned deadline."],
                ],
                inputs=policy_input
            )
            read_btn = gr.Button("Read Policy Aloud", variant="primary")
            policy_audio = gr.Audio(label="Policy explanation", autoplay=True)

            read_btn.click(
                fn=read_policy,
                inputs=policy_input,
                outputs=policy_audio
            )

if __name__ == "__main__":
    app.launch()

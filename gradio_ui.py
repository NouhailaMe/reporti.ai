import gradio as gr

from pipeline import run_pipeline

def chat(question):

    result = run_pipeline(
        question
    )

    return result

demo = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="json",
    title="AI Database Assistant"
)

demo.launch()
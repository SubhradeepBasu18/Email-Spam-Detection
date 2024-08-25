import gradio as gr
import joblib

# Load the trained model
model = joblib.load('model.pkl')

def predict_spam(message):
    prediction = model.predict([message])
    return "Spam" if prediction[0] == 1 else "Not Spam"

# Create and launch the Gradio interface
interface = gr.Interface(
    fn=predict_spam,
    inputs="text",
    outputs="text",
    title="Spam Detector",
    description="Enter a message to check if it's Spam or Not Spam."
)

interface.launch(share=True)

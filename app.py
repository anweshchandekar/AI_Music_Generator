import gradio as gr
from keras.models import load_model
import numpy as np
import midi  # or your preferred MIDI library

# Load the AI Music Generator model
model = load_model('AI_Music_Generator.h5')

def generate_music(input_sequence):
    # Preprocess the input sequence if necessary
    # ...

    # Generate music with the model
    predictions = model.predict(input_sequence)
    
    # Convert predictions to MIDI (this is just a placeholder, you'll need your own conversion logic)
    midi_output = convert_to_midi(predictions)
    
    # Save the MIDI file
    midi_file_path = "generated_music.mid"
    with open(midi_file_path, "wb") as f:
        midi_output.writeFile(f)
    
    return midi_file_path

# Create Gradio interface
interface = gr.Interface(
    fn=generate_music,
    inputs=gr.Textbox(lines=2, placeholder="Enter music sequence here..."),
    outputs=gr.File(label="Download generated MIDI file"),
    title="AI Music Generator"
)

if __name__ == "__main__":
    interface.launch()
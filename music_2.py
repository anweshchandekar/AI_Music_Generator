import os
import numpy as np
from collections import Counter
from music21 import converter, instrument, note, chord, stream
from keras.layers import Embedding, Conv1D, Dropout, MaxPool1D, GlobalMaxPool1D, Dense
from keras.models import Sequential
import keras.backend as K
import matplotlib.pyplot as plt
import random

# Function to read MIDI files and extract notes
def read_midi(file):
    print("Loading Music File:", file)
    
    notes = []
    notes_to_parse = None
    
    # Parsing a MIDI file
    midi = converter.parse(file)
  
    # Grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    # Looping over all the instruments
    for part in s2.parts:
        # Select elements of only piano
        if 'Piano' in str(part): 
            notes_to_parse = part.recurse()
            # Finding whether a particular element is note or a chord
            for element in notes_to_parse:
                # Note
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                # Chord
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    return np.array(notes)

# Specify the path and read the MIDI file
path = r'C:\Users\anwes\OneDrive\Desktop\AI_Music_Generator\only_time_will_tell.mid'
notes = read_midi(path)

# Process notes to create sequences
no_of_timesteps = 32

# Convert notes into input sequences
x = []
for i in range(0, len(notes) - no_of_timesteps, 1):
    input_ = notes[i:i + no_of_timesteps]
    x.append(input_)
x = np.array(x)

# Convert notes to integers for the model
x_note_to_int = dict((note, num) for num, note in enumerate(sorted(set(notes))))
int_to_note = dict((num, note) for note, num in x_note_to_int.items())
x_seq = np.array([[x_note_to_int[note] for note in seq] for seq in x])

# Define the model (assuming it's already trained or will be trained)
K.clear_session()
model = Sequential()
model.add(Embedding(len(x_note_to_int), 100, input_length=no_of_timesteps, trainable=True)) 
model.add(Conv1D(64, 3, padding='causal', activation='relu'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))
model.add(Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))
model.add(Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'))
model.add(Dropout(0.2))
model.add(MaxPool1D(2))
model.add(GlobalMaxPool1D())
model.add(Dense(256, activation='relu'))
model.add(Dense(len(x_note_to_int), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

# Generate predictions
ind = np.random.randint(0, len(x_seq) - 1)
random_music = x_seq[ind]

predictions = []
for _ in range(10):
    random_music = random_music.reshape(1, no_of_timesteps)
    prob = model.predict(random_music)[0]
    y_pred = np.argmax(prob, axis=0)
    predictions.append(y_pred)
    random_music = np.append(random_music[0][1:], y_pred)

print("Predictions:", predictions)

# Convert predictions to notes/chords
predicted_notes = [int_to_note[pred] for pred in predictions]

# Convert predictions to MIDI
def convert_to_midi(prediction_output):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                try:
                    cn = int(current_note)
                    new_note = note.Note(cn)
                    new_note.storedInstrument = instrument.Piano()
                    notes.append(new_note)
                except ValueError:
                    print(f"Invalid note value: {current_note}")
                    continue
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            try:
                new_note = note.Note(pattern)
                new_note.offset = offset
                new_note.storedInstrument = instrument.Piano()
                output_notes.append(new_note)
            except Exception as e:
                print(f"Error creating note {pattern}: {e}")
                continue
        offset += 1

    midi_stream = stream.Stream(output_notes)
    output_path = r'C:\Users\anwes\OneDrive\Desktop\AI_Music_Generator\music.mid'
    
    try:
        midi_stream.write('midi', fp=output_path)
        print(f"MIDI file created successfully at {output_path}")
    except Exception as e:
        print(f"Failed to save MIDI file: {e}")

# Pass the predictions to the conversion function
convert_to_midi(predicted_notes)
model.save(r"C:\Users\anwes\OneDrive\Desktop\AI_Music_Generator\scripts\AI_Music_Generator.h5")

print("Model saved successfully at C:\\Users\\anwes\\OneDrive\\Desktop\\AI_Music_Generator\\scripts\\AI_Music_Generator.h5")
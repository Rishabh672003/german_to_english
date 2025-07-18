# Updated main.py with CORS support
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # Add this import
from pydantic import BaseModel
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load artifacts
with open('eng_tokenizer.pickle', 'rb') as f:
    eng_tokenizer = pickle.load(f)

with open('deu_tokenizer.pickle', 'rb') as f:
    deu_tokenizer = pickle.load(f)

model = load_model('translation_model.h5')

# Constants from training
MAX_LENGTH = 8

class TranslationRequest(BaseModel):
    text: str

def preprocess_text(text):
    # Remove punctuation and lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.lower()

def decode_sequence(input_seq):
    prediction = model.predict(input_seq.reshape(1, MAX_LENGTH))
    indices = np.argmax(prediction[0], axis=-1)
    return ' '.join([eng_tokenizer.index_word.get(idx, '') for idx in indices if idx > 0])

@app.post("/translate")
async def translate(request: TranslationRequest):
    # Preprocess input
    text = preprocess_text(request.text)
    
    # Tokenize and pad
    seq = deu_tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post')
    
    # Translate
    translated = decode_sequence(padded)
    return {"translation": translated}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

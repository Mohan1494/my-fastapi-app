import joblib
import re
import os
from fastapi import FastAPI
from pydantic import BaseModel

# Load model and vectorizer from the current directory
model_directory = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(model_directory, 'sentiment_model.joblib')
vectorizer_path = os.path.join(model_directory, 'vectorizer.joblib')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Create a FastAPI instance
app = FastAPI()

# Define the input data structure
class InputData(BaseModel):
    text: str

# Preprocess text function
def preprocess_text(text):
    """Preprocess text by removing unwanted characters."""
    cleaned_text = re.sub('[^அ-ஹஂா-ு-ூெ-ைொ-்]', ' ', text)
    return cleaned_text

@app.post('/predict/')
async def predict(data: InputData):
    # Preprocess user input
    preprocessed_input = preprocess_text(data.text)

    # Vectorize user input
    input_vectorized = vectorizer.transform([preprocessed_input])

    # Predict sentiment
    prediction = model.predict(input_vectorized)

    # Debugging output
    print(f"Raw Prediction: {prediction[0]}")
    print(f"Model Classes: {model.classes_}")

    # Convert prediction to human-readable format
    sentiment = prediction[0]  # This should now be a string like 'Favorable', 'Neutral', or 'Not Favorable'
    
    # Map model output to desired labels
    sentiment_labels = {
        'Favorable': 'positive',
        'Neutral': 'neutral',
        'Not Favorable': 'negative'
    }
    
    result = sentiment_labels.get(sentiment, "unknown")

    return {'sentiment': result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

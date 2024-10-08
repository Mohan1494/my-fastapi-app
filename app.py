import joblib
import re
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and vectorizer from the current directory
try:
    model_directory = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_directory, 'sentiment_model.joblib')
    vectorizer_path = os.path.join(model_directory, 'vectorizer.joblib')

    # Ensure model files exist
    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError("Model or vectorizer files not found")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # Ensure loaded model and vectorizer are of correct types
    if not hasattr(model, "predict") or not isinstance(vectorizer, TfidfVectorizer):
        raise ValueError("Invalid model or vectorizer loaded")

    logger.info("Model and vectorizer loaded successfully.")

except Exception as e:
    logger.error(f"Error loading model or vectorizer: {e}")
    raise

# Create a FastAPI instance
app = FastAPI()

# CORS configuration to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    try:
        # Preprocess user input
        preprocessed_input = preprocess_text(data.text)

        # Vectorize user input
        input_vectorized = vectorizer.transform([preprocessed_input])

        # Predict sentiment
        prediction = model.predict(input_vectorized)

        # Debugging output
        logger.info(f"Raw Prediction: {prediction[0]}")
        logger.info(f"Model Classes: {model.classes_}")

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
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return {'sentiment': 'error', 'details': str(e)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)

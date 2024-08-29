from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import re
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

app = Flask(__name__)

# Load the saved Random Forest model and TfidfVectorizer
with open('rf_classifier.pkl', 'rb') as file:
    loaded_rf_classifier = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to clean the text
def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the review text from the form
    review_text = request.form.get('review', '')

    # Clean the review text
    cleaned_text = clean_text(review_text)

    # Create a DataFrame for compatibility with TfidfVectorizer
    df = pd.DataFrame([cleaned_text], columns=['cleaned_review_content'])

    # Transform the cleaned text using the loaded vectorizer
    X_vectorized = vectorizer.transform(df['cleaned_review_content'])

    # Predict using the loaded Random Forest model
    prediction = loaded_rf_classifier.predict(X_vectorized)

    # Render the result in the template
    return render_template('index.html', prediction=prediction[0], review=review_text)

if __name__ == '__main__':
    app.run(debug=True)

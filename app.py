from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import warnings
import os
# The 're' import has been removed as it's no longer needed

warnings.filterwarnings("ignore")

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# The clean_text function has been REMOVED as requested.

def load_model(filename):
    """Safely loads a joblib file."""
    if not os.path.exists(filename):
        print(f"Error: Model file not found: {filename}. The server cannot start.")
        return None
    try:
        return joblib.load(filename)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# --- MODEL LOADING ---
print("Loading models and vectorizers...")
v_fake = load_model("fakenews_vectorizer.joblib")
m_fake = load_model("fakenews_model.joblib")
v_cat = load_model("news_category_vector.joblib")
m_cat = load_model("news_category_model.joblib")
v_hate = load_model("hatespeach_vectorizer.joblib")
m_hate = load_model("hatespeach_model.joblib")

if not all([v_fake, m_fake, v_cat, m_cat, v_hate, m_hate]):
    print("One or more core model/vectorizer files are missing. Cannot start the server.")
    exit()

print("Models and vectorizers loaded successfully.")

# --- PREDICTION FUNCTIONS (Using Raw Text) ---
label_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

def predict_fake(txt: str):
    # The raw text 'txt' is used directly
    vectorized_text = v_fake.transform([txt])
    prediction = m_fake.predict(vectorized_text)[0]
    return "Real" if prediction == 1 else "Fake"

def predict_cat(txt: str):
    try:
        # The raw text 'txt' is used directly
        vectorized_text = v_cat.transform([txt])
        prediction_idx = m_cat.predict(vectorized_text)[0]
        return label_map.get(prediction_idx, "Unknown")
    except Exception:
        return "Unknown"

def predict_hate(txt: str):
    # The raw text 'txt' is used directly
    vectorized_text = v_hate.transform([txt])
    prediction = m_hate.predict(vectorized_text)[0]
    return True if prediction == 1 else False

# --- API ENDPOINTS ---
@app.route('/')
def serve_index():
    return send_from_directory('.', 'templates/index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if 'article_text' not in data or not data['article_text'].strip():
            return jsonify({"error": "Article text is missing"}), 400
        
        article_text = data['article_text']

        results = {
            "authenticity": {"prediction": predict_fake(article_text)},
            "category": {"name": predict_cat(article_text)},
            "hateSpeech": {"detected": predict_hate(article_text)}
        }
        return jsonify(results)
    except Exception as e:
        print(f"An error occurred during analysis: {e}")
        return jsonify({"error": "An internal server error occurred"}), 500

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)


from flask import Flask, render_template, request, jsonify
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load your trained model (assuming lr.pkl is in the same directory as app.py)
model_prd = joblib.load('lr.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request (JSON format)
        data = request.get_json()
        cgpa = float(data['cgpa'])
        
        # Predict the package based on CGPA
        prediction = model_prd.predict([[cgpa]])

        # Return prediction as JSON response
        return jsonify({'prediction': prediction[0]})
    
    except Exception as e:
        # Handle any error
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)

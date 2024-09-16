from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load your pre-trained churn model (assuming you have a 'model.pkl' file)
model = joblib.load('model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form (index.html)
        tenure = int(request.form['tenure'])
        num_of_products = int(request.form['num_of_products'])
        has_cr_card = int(request.form['has_cr_card'])
        is_active_member = int(request.form['is_active_member'])
        credit_score = float(request.form['credit_score'])
        age = float(request.form['age'])
        balance = float(request.form['balance'])
        estimated_salary = float(request.form['estimated_salary'])

        # Arrange data as expected by the model
        data = np.array([[credit_score, age, balance, estimated_salary, tenure, num_of_products, has_cr_card, is_active_member]])

        # Make prediction
        prediction = model.predict(data)
        output = 'Customer will exit' if prediction[0] == 1 else 'Customer will stay'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

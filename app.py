import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('classification.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    # Get JSON data from request
    data = request.get_json()  # Get the JSON data from the request body
    print(data)  # For debugging: Check the received data
    
    # Convert input data to a numpy array and reshape for the model
    to_predict = np.array(list(data.values())).reshape(1, -1)
    
    # Predict using the loaded model
    output = model.predict(to_predict)
    print(output[0])  # For debugging: Check the prediction output
    
    # Ensure the output is JSON serializable
    serializable_output = int(output[0]) if isinstance(output[0], np.integer) else float(output[0])
    
    # Return the output as JSON
    return jsonify({'prediction': serializable_output})

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = np.array(data).reshape(1,-1)
    output = model.predict(final_input)[0]
    if output==1:
        final_output = 'YES'
    else:
        final_output = 'NO'
    return render_template("home.html",prediction_text = "The prediction of possible heart disease in future {}".format(final_output))

if __name__ == "__main__":
    app.run(debug=True)

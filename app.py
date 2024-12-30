# # app.py

# from flask import Flask, request, jsonify, render_template
# import pickle
# import numpy as np

# # Load the trained model
# model_path = 'D:\VS-Code\Hypertension Project\hypertension.pkl'
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     # Extract data from form
#     int_features = [int(x) for x in request.form.values()]
#     final_features = np.array(int_features)
#     final_features_list=final_features.tolist()
#     # Make prediction
#     prediction = model.predict(final_features_list)
#     output = 'Positive -{}-'.format(prediction[0]) if prediction[0] == 1 else 'Negative -{}-'.format(prediction[0])

#     return render_template('index.html', prediction_text='Prediction: {}'.format(output))

# if __name__ == "__main__":
#     app.run(debug=True)
# ======================================================================================================================================================
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = 'D:\\VS-Code\\Hypertension Project\\hypertension.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at: {model_path}")

with open(model_path, 'rb') as file:
    model = pickle.load(file)
# with open('hypertension.pkl', 'wb') as file:
#     pickle.dump(model, file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        int_features = [int(x) for x in request.form.values()]
        final_features = np.array(int_features).reshape(1, -1)
        
        # Debugging print statements
        print(f"Input features: {int_features}")
        print(f"Reshaped features: {final_features}")
        print(type(model))
        
        # Make prediction
        prediction = model.predict(final_features)
        output = 'Positive -{}-'.format(prediction[0]) if prediction[0] == 1 else 'Negative -{}-'.format(prediction[0])
        
        return render_template('index.html', prediction_text='Prediction: {}'.format(output))
    except Exception as e:
        # Print the exception error message for debugging
        print(f"Error: {e}")
        return render_template('index.html', prediction_text='Error occurred: {}'.format(e))

if __name__ == "__main__":
    app.run(debug=True)
# ======================================================================================================================================================
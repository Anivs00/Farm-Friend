# import pandas as pd
# from flask import Flask, render_template, request, jsonify
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report  # Import model evaluation metrics
# import os

# app = Flask(__name__)

# # Load the dataset
# data = pd.read_csv("Crop_data.csv")

# # Separate features (X) and labels (y)
# X = data.drop(columns=['label'])
# y = data['label']

# # Create and train a Decision Tree model
# dt_model = DecisionTreeClassifier(random_state=42)
# rf_model = RandomForestClassifier(random_state=42)

# dt_model.fit(X, y)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         # Get user input from the form
#         n = float(request.form['n'])
#         p = float(request.form['p'])
#         k = float(request.form['k'])
#         temperature = float(request.form['temperature'])
#         humidity = float(request.form['humidity'])
#         ph = float(request.form['ph'])
#         rainfall = float(request.form['rainfall'])
#         prediction_mode = float(request.form['prediction_mode'])

#         # Create a data frame from user input
#         user_input = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
#                                     columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
#         print("User Input Data:", user_input)
        
#         if prediction_mode == 'decision_tree':
#             # Use the Decision Tree model
#             prediction = dt_model.predict(user_input)
#             print("Training Accuracy: {accuracy_train}")
#         else:
#             # Use the Random Forest model
#             prediction = rf_model.predict(user_input)
#             print("Training Accuracy: {accuracy_train}")
            
        


#         # Model Evaluation (Add this code here)
#         y_pred_train = dt_model.predict(X)

#         # Evaluate model performance on training data
#         accuracy_train = accuracy_score(y, y_pred_train)
#         report_train = classification_report(y, y_pred_train)

#         # print("Training Accuracy: {accuracy_train}")
#         print("Classification Report (Training):\n", report_train)

# # Return the prediction result as JSON response
#         response = {
#             'model_choice': model_choice,
#             'prediction': prediction[0]
#         }
#         return jsonify(response)
# if __name__ == '__main__':
#     app.run(debug=True)

import pandas as pd
from flask import Flask, render_template, request ,jsonify
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("Crop_data.csv")

# Separate features (X) and labels (y)
X = data.drop(columns=['label'])
y = data['label']

# Create and train a Decision Tree and a Random Forest model
# dt_model = DecisionTreeClassifier(random_state=42)
# rf_model = RandomForestClassifier(random_state=42)
# dt_model.fit(X, y)
# rf_model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input from the form
        n = float(request.form['n'])
        p = float(request.form['p'])
        k = float(request.form['k'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        model_choice = request.form['model_choice']
        

        # Create a data frame from user input
        user_input = pd.DataFrame([[n, p, k, temperature, humidity, ph, rainfall]],
                                  columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
        
        if model_choice == 'decision_tree':
            # Use the Decision Tree model
            # prediction = dt_model.predict(user_input)
            # accuracy = accuracy_score(y, dt_model.predict(X))
            model = DecisionTreeClassifier(random_state=42)
        else:
            # Use the Random Forest model
            # prediction = rf_model.predict(user_input)
            # accuracy = accuracy_score(y, rf_model.predict(X))
            model = RandomForestClassifier(random_state=42)
        
    model.fit(X, y)
    prediction = model.predict(user_input)
    accuracy = accuracy_score(y, model.predict(X))
    
        
    # Return the prediction result and accuracy as a JSON response
    # Create a JSON response
    
    # def get_json_data():
    response = {
                'model_choice': model_choice,
                'prediction': prediction[0],
                'accuracy': accuracy
            }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
        # return render_template('result.html', **response)

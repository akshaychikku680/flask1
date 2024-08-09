from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and column information
with open('total_alcohol_model.pkl', 'rb') as file:
    model, columns=pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict total alcohol consumption based on the input features."""

    beer_servings=float(request.form['beer_servings'])
    spirit_servings=float(request.form['spirit_servings'])
    wine_servings=float(request.form['wine_servings'])
    continent=request.form['continent']


    input_df = pd.DataFrame({
        'beer_servings': [beer_servings],
        'spirit_servings': [spirit_servings],
        'wine_servings': [wine_servings],

        'continent': [continent]
    })


    input_encoded=pd.get_dummies(input_df, columns=['continent'])

# Reindex to match the model columns
    input_encoded=input_encoded.reindex(columns=columns, fill_value=0)


    input_features=input_encoded.values


    prediction=model.predict(input_features)[0]

    return render_template('result.html', prediction=prediction)

if __name__=='__main__':
    app.run(debug=True)

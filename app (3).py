from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained pipeline (StandardScaler + RandomForestClassifier)
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Feature order must match the order used during training
FEATURES = ['Gender', 'Age', 'EstimatedSalary']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get and validate user input from the form
    try:
        gender = int(request.form['gender'])              # 1 = Male, 0 = Female
        age = int(request.form['age'])
        estimated_salary = int(request.form['estimated_salary'])
    except (KeyError, ValueError):
        return render_template(
            'result.html',
            result="could not be predicted because the input was invalid"
        )

    # Build a DataFrame with the same column names and order as training
    input_data = pd.DataFrame([[gender, age, estimated_salary]], columns=FEATURES)

    # The pipeline scales the input automatically before predicting
    prediction = model.predict(input_data)
    result = "will purchase the product" if prediction[0] == 1 else "will not purchase the product"

    return render_template('result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)

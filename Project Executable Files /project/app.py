import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

model1 = pickle.load(open('productivity.pkl', 'rb'))

app = Flask(__name__)
app.static_folder = 'templates/static'  


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def index():
    return render_template('predict.html')

@app.route('/data_predict', methods=['POST'])
def predict():
    try:
        # Get form data
        quarter = int(request.form['Quarter'])
        department = request.form['Department']
        if department.lower() == 'sewing':
            department = 1
        else:
            department = 0

        day = request.form['Day of the week']
        day_mapping = {'Monday': 0, 'Tuesday': 4, 'Wednesday': 5, 'Thursday': 3, 'Saturday': 1, 'Sunday': 2, 'Friday': 6}
        day = day_mapping.get(day, -1)  # default to -1 if not found

        team = int(request.form['Team Number'])
        time = float(request.form['Time Allocated'])
        items = int(request.form['Unfinished Items'])
        over_time = float(request.form['Over time'])
        incentive = int(request.form['Incentive'])
        idle_time = int(request.form['Idle Time'])
        idle_men = int(request.form['Idle Men'])
        style = int(request.form['Style Change'])
        workers = int(request.form['Number of Workers'])
        targeted_productivity = float(request.form['targeted productivity'])

        if not 0 <= targeted_productivity <= 1:
            return "Error: Targeted productivity must be between 0 and 1."

        input_data = [quarter, department, day, team, time, items, over_time, incentive, idle_time, idle_men, style, workers, targeted_productivity]
        columns = ['quarter', 'department', 'day', 'team', 'time', 'items', 'over_time', 'incentive', 'idle_time', 'idle_men', 'style', 'workers', 'targeted_productivity']

        print(f"Received data - {input_data}")
        
        data = pd.DataFrame([input_data], columns=columns)
        print(data)  # Debug: print DataFrame to verify

        prediction = model1.predict(data)
        prediction = np.clip(prediction, 0, 1)

        prediction_percentage = prediction[0] * 100

        print(f"Prediction: {prediction_percentage:.2f}%")

        return render_template('productivity.html', prediction_text=f"Productivity prediction is {prediction_percentage:.2f}%")

    except Exception as e:
        return f"Error: {e}. Please ensure all fields are filled out correctly."

if __name__ == '__main__':
    app.run(debug=True)

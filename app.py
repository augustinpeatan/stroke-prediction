from flask import Flask, render_template, request
import pickle
import numpy as np



app = Flask(__name__)
loaded_model = pickle.load(open('rf model.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prediction", methods=['POST', 'GET'])
def predict():
    gender =int(request.form['gender'])
    age =int(request.form['age'])
    hypertension =int(request.form['hypertension'])
    heart_disease =int(request.form['heart_disease'])
    ever_married =int(request.form['ever_married'])
    work_type =int(request.form['work_type'])
    Residence_type =int(request.form['Residence_type'])
    avg_glucose_level =float(request.form['avg_glucose_level'])
    bmi =float(request.form['bmi'])
    smoking_status =int(request.form['smoking_status'])

    prediction = loaded_model.predict([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]])
    probability = loaded_model.predict([[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status]])
    probability = f"{np.round((np.max(probability)* 100), 2)}%"

    if prediction [0] == 0:
        prediction = "No Stroke"

    else:
        prediction = "Stroke"

    return render_template("index.html", output_prediction=prediction, output_proba=probability)




if __name__ == "__main__":
    app.run(debug=True )

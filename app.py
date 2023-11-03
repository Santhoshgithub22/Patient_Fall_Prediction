from flask import Flask, render_template, jsonify, request
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        data=CustomData(
            acc_max = float(request.form.get('acc_max')),
            gyro_max = float(request.form.get('gyro_max')),
            acc_kurtosis= float(request.form.get('kurtosis')),
            gyro_kurtosis = float(request.form.get('gyro_kurtosis')),
            #label = request.form.get('label'),
            lin_max = float(request.form.get('lin_max')),
            acc_skewness = float(request.form.get('skewness')),
            gyro_skewness = float(request.form.get('gyro_skewness')),
            post_gyro_max = float(request.form.get('post_gyro_max')),
            post_lin_max = float(request.form('post_lin_max')),
            fall = float(request.form('fall'))
        )

        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        if pred==0:
            results = "NOT FALL"
        else:
            results = "FALL"

        return render_template('results.html', final_result = results)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
from flask import Flask, render_template, jsonify, request
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.logger import logging

app = Flask(__name__)

#logging.info("First Route Has Started")
@app.route("/")
def home_page():
    return render_template('index.html')
logging.info("First Route Has Been Completed")

#logging.info("Second Route Has Started")
@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        post_gyro_max = request.form.get('post_gyro_max')
        if post_gyro_max is not None and post_gyro_max.strip() != "":
            post_gyro_max = float(post_gyro_max)
        else:
            post_gyro_max = 0.0  # Provide a default value

        data = CustomData(
            acc_max=float(request.form.get('acc_max')),
            gyro_max=float(request.form.get('gyro_max')),
            acc_kurtosis=float(request.form.get('acc_kurtosis')),
            gyro_kurtosis=float(request.form.get('gyro_kurtosis')),
            lin_max=float(request.form.get('lin_max')),
            acc_skewness=float(request.form.get('acc_skewness')),
            gyro_skewness=float(request.form.get('gyro_skewness')),
            post_gyro_max=post_gyro_max,
            post_lin_max=float(request.form.get("post_lin_max"))
        )

        logging.info(f"Data is {data}")
        final_new_data = data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        pred = predict_pipeline.predict(final_new_data)

        if pred == 0:
            results = "NOT FALL"
        else:
            results = "FALL"
        
        logging.info("Second Route Has Completed")
        return render_template('index.html', final_result=results)
    

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)

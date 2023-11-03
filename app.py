from flask import Flask, render_template, jsonify
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
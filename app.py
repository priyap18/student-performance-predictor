from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model safely
try:
    model_path = os.path.join(os.path.dirname(__file__), "MLR.pkl")

    with open(model_path, "rb") as file:
        model = pickle.load(file)

except Exception as e:
    print("Error loading model:", e)
    model = None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return render_template(
            "index.html",
            prediction_text="Model could not be loaded."
        )

    try:

        hours_studied = float(request.form["hours_studied"])
        previous_scores = float(request.form["previous_scores"])
        extracurricular = int(request.form["extracurricular"])
        sleep_hours = float(request.form["sleep_hours"])
        sample_papers = float(request.form["sample_papers"])

        sample = np.array([[
            hours_studied,
            previous_scores,
            extracurricular,
            sleep_hours,
            sample_papers
        ]])

        prediction = model.predict(sample)

        predicted_score = round(float(prediction[0]), 2)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Performance Index: {predicted_score}"
        )

    except ValueError:
        return render_template(
            "index.html",
            prediction_text="Please enter valid numeric values."
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

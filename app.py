from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load models
reg_model = joblib.load("models/regression_model.pkl")

# Load all classification models separately
logistic_model = joblib.load("models/logistic_model.pkl")
decision_tree_model = joblib.load("models/decision_tree_model.pkl")
random_forest_model = joblib.load("models/classification_model.pkl")

kmeans_model = joblib.load("models/kmeans_model.pkl")

scaler = joblib.load("models/scaler.pkl")
cluster_scaler = joblib.load("models/cluster_scaler.pkl")


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":

        # Get form values
        features = [
            float(request.form["MedInc"]),
            float(request.form["HouseAge"]),
            float(request.form["AveRooms"]),
            float(request.form["AveBedrms"]),
            float(request.form["Population"]),
            float(request.form["AveOccup"]),
            float(request.form["Latitude"]),
            float(request.form["Longitude"])
        ]

        # Get selected classification model
        selected_model = request.form["model"]

        features_array = np.array([features])

        # Scale for regression & classification
        features_scaled = scaler.transform(features_array)

        # ---------------- Regression ----------------
        predicted_price = reg_model.predict(features_scaled)[0]

        # ---------------- Classification ----------------
        if selected_model == "logistic":
            model_used = logistic_model
            model_name = "Logistic Regression"
        elif selected_model == "decision_tree":
            model_used = decision_tree_model
            model_name = "Decision Tree"
        elif selected_model == "random_forest":
            model_used = random_forest_model
            model_name = "Random Forest"
        else:
            model_used = logistic_model
            model_name = "Logistic Regression"

        class_prediction = model_used.predict(features_scaled)[0]

        if class_prediction == 0:
            predicted_class = "Low"
        elif class_prediction == 1:
            predicted_class = "Medium"
        else:
            predicted_class = "High"

        # ---------------- Clustering ----------------
        cluster_scaled = cluster_scaler.transform(features_array)
        cluster_number = kmeans_model.predict(cluster_scaled)[0]

        cluster_mapping = {
            0: "Affluent Coastal Residential Zone",
            1: "Mid-Income Urban Density Zone",
            2: "Emerging Inland Residential Area"
        }

        predicted_cluster = cluster_mapping.get(cluster_number, f"Region {cluster_number}")

        return render_template(
            "predict.html",
            predicted_price=round(predicted_price * 100000, 2),
            predicted_class=predicted_class,
            predicted_cluster=predicted_cluster,
            model_name=model_name
        )

    return render_template("predict.html")


if __name__ == "__main__":
    app.run(debug=True)
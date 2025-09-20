from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")  # input form

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Collect inputs from form
        car_name = request.form["car_name"]
        yr_mfr = int(request.form["yr_mfr"])
        fuel_type = request.form["fuel_type"]
        kms_run = float(request.form["kms_run"])
        sale_price = float(request.form["sale_price"])
        city = request.form["city"]
        times_viewed = int(request.form["times_viewed"])
        body_type = request.form["body_type"]
        transmission = request.form["transmission"]
        variant = request.form["variant"]
        assured_buy = request.form["assured_buy"]
        registered_city = request.form["registered_city"]
        registered_state = request.form["registered_state"]
        is_hot = request.form["is_hot"]
        rto = request.form["rto"]
        source = request.form["source"]
        make = request.form["make"]
        model_name = request.form["model"]
        car_availability = request.form["car_availability"]
        total_owners = int(request.form["total_owners"])
        broker_quote = float(request.form["broker_quote"])
        original_price = float(request.form["original_price"])
        car_rating = float(request.form["car_rating"])
        ad_created_on = request.form["ad_created_on"]
        fitness_certificate = request.form["fitness_certificate"]
        emi_starts_from = float(request.form["emi_starts_from"])
        booking_down_pymnt = float(request.form["booking_down_pymnt"])
        reserved = request.form["reserved"]
        warranty_avail = request.form["warranty_avail"]

        # Put into dataframe (adjust columns as per your training dataset)
        input_data = pd.DataFrame([[
            car_name, yr_mfr, fuel_type, kms_run, sale_price, city,
            times_viewed, body_type, transmission, variant, assured_buy,
            registered_city, registered_state, is_hot, rto, source, make,
            model_name, car_availability, total_owners, broker_quote,
            original_price, car_rating, ad_created_on, fitness_certificate,
            emi_starts_from, booking_down_pymnt, reserved, warranty_avail
        ]], columns=[
            "car_name", "yr_mfr", "fuel_type", "kms_run", "sale_price", "city",
            "times_viewed", "body_type", "transmission", "variant", "assured_buy",
            "registered_city", "registered_state", "is_hot", "rto", "source", "make",
            "model", "car_availability", "total_owners", "broker_quote",
            "original_price", "car_rating", "ad_created_on", "fitness_certificate",
            "emi_starts_from", "booking_down_pymnt", "reserved", "warranty_avail"
        ])

        # Predict
        prediction = model.predict(input_data)[0]

        return render_template("index.html", prediction_text=f"Predicted Value: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
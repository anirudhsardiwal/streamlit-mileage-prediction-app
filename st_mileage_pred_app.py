import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pickle

df = pd.read_csv("auto-mpg.csv")

df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
df["horsepower"] = df["horsepower"].replace("?", np.nan)
df["horsepower"] = df["horsepower"].fillna(df["horsepower"].median())
df["make"] = df["car name"].apply(lambda x: x.split()[0])
df["make"] = df["make"].replace(
    {
        "chevy": "chevrolet",
        "chevroelt": "chevrolet",
        "maxda": "mazda",
        "mercedes-benz": "mercedes",
        "toyouta": "toyota",
        "vokswagen": "volkswagen",
        "vw": "volkswagen",
    }
)
df = df.drop("car name", axis=1)
df["origin"] = df["origin"].replace({1: "america", 2: "europe", 3: "asia"})
dfEncoded = pd.get_dummies(df, drop_first=True)
dfEncoded = dfEncoded.replace({False: 0, True: 1})
X = dfEncoded.drop("mpg", axis=1)
y = dfEncoded["mpg"]

with st.form("main form", enter_to_submit=False):
    cylinders = st.selectbox("No. of Cylinders", [3, 4, 5, 6, 8])
    displacement = st.number_input("Displacement (cc)", 0, value=190)
    horsepower = st.number_input("Horsepower (bhp)", 0, value=100)
    weight = st.number_input("Weight (kg)", 0, value=2800)
    model_year = st.number_input("Year (between '75 and '82)", 75, value=80)
    acceleration = st.number_input("Acceleration (m/s2)", 0, value=12)
    origin = st.selectbox("Origin", ["America", "Asia", "Europe"])
    make = st.selectbox(
        "Make",
        [
            "Amc",
            "Audi",
            "Bmw",
            "Buick",
            "Cadillac",
            "Capri",
            "Chevrolet",
            "Chrysler",
            "Datsun",
            "Dodge",
            "Fiat",
            "Ford",
            "Hi",
            "Honda",
            "Mazda",
            "Mercedes",
            "Mercury",
            "Nissan",
            "Oldsmobile",
            "Opel",
            "Peugeot",
            "Plymouth",
            "Pontiac",
            "Renault",
            "Saab",
            "Subaru",
            "Toyota",
            "Triumph",
            "Volkswagen",
            "Volvo",
        ],
    )
    submitted = st.form_submit_button()

input_data = {
    "cylinders": cylinders,
    "displacement": displacement,
    "horsepower": horsepower,
    "weight": weight,
    "acceleration": acceleration,
    "model year": model_year,
    "origin": origin,
    "make": make,
}

to_predict_list = list(input_data.values())

to_predict_list[:-2] = list(map(int, to_predict_list[:-2]))

origin_pred = to_predict_list[-2]
make_pred = to_predict_list[-1]

origin_mapping = {"America": 0, "Asia": 1, "Europe": 2}

make_mapping = {
    "Amc": 0,
    "Audi": 1,
    "Bmw": 2,
    "Buick": 3,
    "Cadillac": 4,
    "Capri": 5,
    "Chevrolet": 6,
    "Chrysler": 7,
    "Datsun": 8,
    "Dodge": 9,
    "Fiat": 10,
    "Ford": 11,
    "Hi": 12,
    "Honda": 13,
    "Mazda": 14,
    "Mercedes": 15,
    "Mercury": 16,
    "Nissan": 17,
    "Oldsmobile": 18,
    "Opel": 19,
    "Peugeot": 20,
    "Plymouth": 21,
    "Pontiac": 22,
    "Renault": 23,
    "Saab": 24,
    "Subaru": 25,
    "Toyota": 26,
    "Triumph": 27,
    "Volkswagen": 28,
    "Volvo": 29,
}

origin_encoded = [0] * (len(origin_mapping) - 1)
make_encoded = [0] * (len(make_mapping) - 1)

if origin_pred in origin_mapping:
    origin_index = origin_mapping[origin_pred] - 1
    if origin_index >= 0:
        origin_encoded[origin_index] = 1

if make_pred in make_mapping:
    make_index = make_mapping[make_pred] - 1
    if make_index >= 0:
        make_encoded[make_index] = 1

to_predict_list = to_predict_list[:-2] + origin_encoded + make_encoded

to_predict_array = np.array(to_predict_list)
to_predict_array = to_predict_array.reshape(1, -1)

with open("mileage_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

result = model.predict(to_predict_array)
result = result.item()

if submitted:
    st.success(
        f"The predicted mileage is **{result:.1f}** mpg with a ± 2.1 mpg accuracy"
    )

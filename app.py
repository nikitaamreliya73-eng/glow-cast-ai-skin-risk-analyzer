from flask import Flask, render_template, request, session
import joblib
import pandas as pd
import os
import requests

from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
try:
    import tensorflow as tf
except:
    tf = None


# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# Required for Flask session
app.secret_key = os.getenv(
    "FLASK_SECRET_KEY",
    "glow-cast-secret-key"
)


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)


# =========================================================
# LOAD EXISTING WEATHER + SKIN RISK MODEL
# =========================================================

model = joblib.load(
    "model/skin_risk_model.pkl"
)

encoder = joblib.load(
    "model/skin_type_encoder.pkl"
)

print(
    "Weather + Skin Risk Model Loaded Successfully!"
)


# =========================================================
# LOAD GLOW CAST CNN V6
# =========================================================

CNN_MODEL_PATH = (
    "model/best_skin_type_cnn_v6.keras"
)

skin_model = tf.keras.models.load_model(
    CNN_MODEL_PATH,
    compile=False
)

print(
    "GLOW CAST CNN V6 Loaded Successfully!"
)


# =========================================================
# SKIN CLASSES
# =========================================================

SKIN_CLASSES = [
    "dry",
    "normal",
    "oily"
]


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# CNN SKIN TYPE PREDICTION
# =========================================================

def predict_skin_type(image_file):

    image = Image.open(
        image_file
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    probabilities = skin_model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_skin_type = (
        SKIN_CLASSES[predicted_index]
    )

    confidence = round(
        float(
            probabilities[predicted_index]
        ) * 100,
        2
    )

    return (
        predicted_skin_type,
        confidence,
        probabilities
    )


# =========================================================
# LIVE WEATHER FROM USER LOCATION
# =========================================================

def get_live_weather(latitude, longitude):

    if not OPENWEATHER_API_KEY:

        print(
            "OpenWeather API Error: "
            "OPENWEATHER_API_KEY is missing from .env"
        )

        return None

    try:

        # -------------------------------------------------
        # Current Weather
        # -------------------------------------------------

        weather_url = (
            "https://api.openweathermap.org/data/2.5/weather"
        )

        params = {

            "lat": latitude,

            "lon": longitude,

            "appid": OPENWEATHER_API_KEY,

            "units": "metric"
        }

        response = requests.get(
            weather_url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        weather_data = response.json()


        # -------------------------------------------------
        # Weather Information
        # -------------------------------------------------

        temperature = float(
            weather_data["main"]["temp"]
        )

        humidity = float(
            weather_data["main"]["humidity"]
        )

        wind_speed = float(
            weather_data["wind"]["speed"]
        )

        weather_condition = (
            weather_data["weather"][0]["description"]
        )


        # -------------------------------------------------
        # City
        # -------------------------------------------------

        detected_city = weather_data.get(
            "name",
            "Unknown Location"
        )

        city_normalization = {

            "Utrān": "Surat",

            "Utran": "Surat",

            "Utrān, Surat": "Surat",

            "Vesu": "Surat",

            "Adajan": "Surat",

            "Varachha": "Surat",

            "Katargam": "Surat",

            "Athwa": "Surat",

            "Pal": "Surat",

            "Piplod": "Surat",

            "Dumas": "Surat",

            "Althan": "Surat",

            "City Light": "Surat",

            "Nanpura": "Surat",

            "Rander": "Surat"
        }

        city = city_normalization.get(
            detected_city,
            detected_city
        )


        # -------------------------------------------------
        # Coordinates returned by OpenWeather
        # -------------------------------------------------

        actual_latitude = (
            weather_data.get(
                "coord",
                {}
            ).get(
                "lat",
                latitude
            )
        )

        actual_longitude = (
            weather_data.get(
                "coord",
                {}
            ).get(
                "lon",
                longitude
            )
        )


        # -------------------------------------------------
        # UV Index
        # -------------------------------------------------

        uv_index = 5.0

        try:

            one_call_url = (
                "https://api.openweathermap.org/data/3.0/onecall"
            )

            one_call_params = {

                "lat": actual_latitude,

                "lon": actual_longitude,

                "appid": OPENWEATHER_API_KEY,

                "exclude":
                    "minutely,hourly,daily,alerts",

                "units": "metric"
            }

            uv_response = requests.get(
                one_call_url,
                params=one_call_params,
                timeout=10
            )

            if uv_response.ok:

                uv_data = uv_response.json()

                uv_index = float(
                    uv_data.get(
                        "current",
                        {}
                    ).get(
                        "uvi",
                        5.0
                    )
                )

        except Exception as uv_error:

            print(
                "UV API unavailable:",
                uv_error
            )


        # -------------------------------------------------
        # Print Live Weather
        # -------------------------------------------------

        print(
            "\n=============================="
        )

        print(
            "GLOW CAST LIVE WEATHER"
        )

        print(
            "=============================="
        )

        print(
            "Detected City:",
            city
        )

        print(
            "Temperature:",
            temperature,
            "°C"
        )

        print(
            "Humidity:",
            humidity,
            "%"
        )

        print(
            "Weather:",
            weather_condition
        )

        print(
            "Wind Speed:",
            wind_speed,
            "m/s"
        )

        print(
            "UV Index:",
            uv_index
        )


        return {

            "city": city,

            "temperature":
                temperature,

            "humidity":
                humidity,

            "weather_condition":
                weather_condition,

            "wind_speed":
                wind_speed,

            "uv_index":
                uv_index
        }


    except requests.exceptions.RequestException as error:

        print(
            "OpenWeather API Error:",
            error
        )

        return None


    except Exception as error:

        print(
            "Weather processing error:",
            error
        )

        return None


# =========================================================
# GLOW CAST RECOMMENDATION ENGINE
# =========================================================

def generate_recommendations(
    skin_type,
    temperature,
    humidity,
    uv_index,
    weather_condition
):

    skin = str(
        skin_type
    ).lower().strip()

    try:

        temperature = float(
            temperature
        )

    except:

        temperature = 30.0

    try:

        humidity = float(
            humidity
        )

    except:

        humidity = 60.0

    try:

        uv_index = float(
            uv_index
        )

    except:

        uv_index = 5.0


    weather = str(
        weather_condition
    ).lower()


    # =====================================================
    # 01 — HYDRATION / MOISTURIZER
    # =====================================================

    if skin == "oily":

        if humidity >= 75:

            moisturizer = (
                "Lightweight Gel Moisturizer"
            )

            hydration_reason = (
                "A lightweight, non-greasy formula "
                "helps maintain hydration without "
                "adding heaviness in humid conditions."
            )

        elif temperature >= 32:

            moisturizer = (
                "Oil-Free Water-Gel Moisturizer"
            )

            hydration_reason = (
                "A water-based moisturizer supports "
                "hydration while keeping the skin feeling fresh."
            )

        else:

            moisturizer = (
                "Lightweight Oil-Free Moisturizer"
            )

            hydration_reason = (
                "A balanced lightweight moisturizer "
                "supports daily hydration for oily skin."
            )


    elif skin == "dry":

        if humidity < 50:

            moisturizer = (
                "Rich Ceramide Moisturizer"
            )

            hydration_reason = (
                "Dry atmospheric conditions can increase "
                "the need for richer moisture support."
            )

        else:

            moisturizer = (
                "Hydrating Barrier Gel-Cream"
            )

            hydration_reason = (
                "A nourishing gel-cream provides hydration "
                "while remaining comfortable in moderate humidity."
            )


    else:

        if humidity >= 75:

            moisturizer = (
                "Lightweight Hydrating Gel-Cream"
            )

            hydration_reason = (
                "A breathable gel-cream keeps normal skin "
                "hydrated without feeling heavy."
            )

        else:

            moisturizer = (
                "Balanced Daily Moisturizer"
            )

            hydration_reason = (
                "A balanced moisturizer supports everyday "
                "hydration for normal skin."
            )


    # =====================================================
    # 02 — SUNSCREEN / PROTECTION
    # =====================================================

    if uv_index >= 7:

        sunscreen = (
            "Broad-Spectrum SPF 50+ Sunscreen"
        )

        protection_reason = (
            "Higher UV conditions call for strong daily "
            "broad-spectrum sun protection."
        )

    elif uv_index >= 3:

        sunscreen = (
            "Broad-Spectrum SPF 50 Sunscreen"
        )

        protection_reason = (
            "Current UV levels make daily broad-spectrum "
            "sun protection an important part of the routine."
        )

    else:

        sunscreen = (
            "Broad-Spectrum SPF 30 Sunscreen"
        )

        protection_reason = (
            "Daily sunscreen helps maintain consistent "
            "UV protection even when UV levels are lower."
        )


    # =====================================================
    # 03 — COMPLEXION / MAKEUP
    # =====================================================

    if skin == "oily":

        if humidity >= 75:

            makeup = (
                "Oil-Control Matte Foundation"
            )

            makeup_reason = (
                "A lightweight matte complexion product "
                "is better suited to oily skin in humid weather."
            )

        elif temperature >= 32:

            makeup = (
                "Long-Wear Oil-Control Foundation"
            )

            makeup_reason = (
                "A breathable oil-control formula can help "
                "maintain a comfortable finish in warmer conditions."
            )

        else:

            makeup = (
                "Natural Matte Foundation"
            )

            makeup_reason = (
                "A soft matte finish complements oily skin "
                "while keeping the complexion lightweight."
            )


    elif skin == "dry":

        if humidity < 50:

            makeup = (
                "Hydrating Dewy Foundation"
            )

            makeup_reason = (
                "A hydrating complexion formula helps support "
                "a comfortable, luminous finish in drier conditions."
            )

        else:

            makeup = (
                "Hydrating Natural-Finish Foundation"
            )

            makeup_reason = (
                "A hydrating natural finish works well "
                "with dry skin in moderate atmospheric conditions."
            )


    else:

        if humidity >= 75:

            makeup = (
                "Lightweight Natural-Matte Foundation"
            )

            makeup_reason = (
                "A lightweight natural-matte finish helps "
                "keep the complexion comfortable in humidity."
            )

        else:

            makeup = (
                "Natural-Finish Foundation"
            )

            makeup_reason = (
                "A natural finish provides balanced everyday "
                "coverage for normal skin."
            )


    # =====================================================
    # 04 — DAILY RITUAL
    # =====================================================

    if skin == "oily":

        if humidity >= 75:

            daily_ritual = (
                "Cleanse gently → hydrate lightly → "
                "apply SPF → use lightweight matte makeup"
            )

        else:

            daily_ritual = (
                "Cleanse → lightweight moisturizer → "
                "SPF → oil-control complexion"
            )


    elif skin == "dry":

        if humidity < 50:

            daily_ritual = (
                "Gentle cleanse → rich hydration → "
                "SPF → hydrating complexion"
            )

        else:

            daily_ritual = (
                "Gentle cleanse → barrier hydration → "
                "SPF → natural-finish makeup"
            )


    else:

        daily_ritual = (
            "Cleanse → balanced hydration → "
            "SPF → natural complexion"
        )


    # =====================================================
    # 05 — BEAUTY INTELLIGENCE
    # =====================================================

    intelligence_points = []


    # Humidity
    if humidity >= 80:

        intelligence_points.append(
            "High humidity detected — keep your routine "
            "lightweight and breathable."
        )

    elif humidity >= 60:

        intelligence_points.append(
            "Moderate-to-high humidity detected — "
            "prioritize comfortable, lightweight hydration."
        )

    elif humidity < 40:

        intelligence_points.append(
            "Low humidity detected — give extra attention "
            "to hydration and skin barrier support."
        )

    else:

        intelligence_points.append(
            "Atmospheric moisture is relatively balanced "
            "for a comfortable daily routine."
        )


    # Temperature
    if temperature >= 35:

        intelligence_points.append(
            "High temperature detected — favor lightweight "
            "textures and avoid overly heavy layers."
        )

    elif temperature >= 30:

        intelligence_points.append(
            "Warm conditions detected — breathable layers "
            "can help keep the routine comfortable."
        )

    elif temperature < 18:

        intelligence_points.append(
            "Cool conditions detected — richer hydration "
            "may help maintain comfort."
        )


    # UV
    if uv_index >= 7:

        intelligence_points.append(
            "UV exposure is elevated — make sun protection "
            "a priority during outdoor exposure."
        )

    elif uv_index >= 3:

        intelligence_points.append(
            "UV exposure is present — maintain consistent "
            "daily sunscreen use."
        )

    else:

        intelligence_points.append(
            "UV levels are currently lower, but daily "
            "sun protection remains part of a consistent routine."
        )


    # Weather
    if (
        "rain" in weather
        or "drizzle" in weather
        or "thunderstorm" in weather
    ):

        intelligence_points.append(
            "Moist or rainy conditions detected — "
            "lighter layers may feel more comfortable."
        )

    elif (
        "cloud" in weather
        or "overcast" in weather
    ):

        intelligence_points.append(
            "Cloudy conditions detected — continue your "
            "regular daytime protection routine."
        )


    beauty_intelligence = " ".join(
        intelligence_points
    )


    # =====================================================
    # RETURN ALL RECOMMENDATIONS
    # =====================================================

    return {

        "moisturizer":
            moisturizer,

        "hydration_reason":
            hydration_reason,

        "sunscreen":
            sunscreen,

        "protection_reason":
            protection_reason,

        "makeup":
            makeup,

        "makeup_reason":
            makeup_reason,

        "daily_ritual":
            daily_ritual,

        "beauty_intelligence":
            beauty_intelligence
    }


# =========================================================
# ANALYZE
# =========================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():


    # =====================================================
    # PHOTO
    # =====================================================

    image_file = (

        request.files.get("photo")

        or request.files.get("image")

        or request.files.get("file")
    )


    if image_file and image_file.filename:

        filename = secure_filename(
            image_file.filename
        )


        print(
            "\n=============================="
        )

        print(
            "GLOW CAST AI SKIN ANALYSIS"
        )

        print(
            "=============================="
        )


        print(
            "Uploaded Image:",
            filename
        )


        # -------------------------------------------------
        # CNN Prediction
        # -------------------------------------------------

        skin_type, skin_confidence, probabilities = (
            predict_skin_type(
                image_file
            )
        )


        print(
            "Predicted Skin Type:",
            skin_type
        )


        print(
            "AI Confidence:",
            skin_confidence,
            "%"
        )


        print(
            "\nProbabilities:"
        )


        print(
            "DRY    :",
            round(
                float(
                    probabilities[0]
                ) * 100,
                2
            ),
            "%"
        )


        print(
            "NORMAL :",
            round(
                float(
                    probabilities[1]
                ) * 100,
                2
            ),
            "%"
        )


        print(
            "OILY   :",
            round(
                float(
                    probabilities[2]
                ) * 100,
                2
            ),
            "%"
        )


    else:

        skin_type = request.form.get(
            "skin_type",
            "normal"
        )

        skin_confidence = None


        print(
            "\nNo image uploaded."
        )


        print(
            "Using selected skin type:",
            skin_type
        )


    # =====================================================
    # USER LOCATION
    # =====================================================

    latitude = request.form.get(
        "latitude"
    )

    longitude = request.form.get(
        "longitude"
    )


    print(
        "\n=============================="
    )

    print(
        "GLOW CAST USER LOCATION"
    )

    print(
        "=============================="
    )


    print(
        "Latitude:",
        latitude
    )


    print(
        "Longitude:",
        longitude
    )


    # =====================================================
    # LIVE WEATHER
    # =====================================================

    weather = None


    if latitude and longitude:

        try:

            latitude = float(
                latitude
            )

            longitude = float(
                longitude
            )


            weather = get_live_weather(
                latitude,
                longitude
            )


        except ValueError:

            print(
                "Invalid location coordinates."
            )


    # =====================================================
    # WEATHER FALLBACK
    # =====================================================

    if weather:

        city = weather["city"]

        temperature = weather[
            "temperature"
        ]

        humidity = weather[
            "humidity"
        ]

        uv_index = weather[
            "uv_index"
        ]

        wind_speed = weather[
            "wind_speed"
        ]

        weather_condition = weather[
            "weather_condition"
        ]


    else:

        print(
            "Using fallback weather values."
        )


        city = "Unknown Location"

        temperature = 30.0

        humidity = 60.0

        uv_index = 5.0

        wind_speed = 10.0

        weather_condition = (
            "Weather unavailable"
        )


    # =====================================================
    # EXISTING RISK MODEL
    # =====================================================

    exposure_hours = float(
        request.form.get(
            "exposure_hours",
            2
        )
    )


    sunscreen_used = int(
        request.form.get(
            "sunscreen_used",
            0
        )
    )


    moisturizer_used = int(
        request.form.get(
            "moisturizer_used",
            0
        )
    )


    skin_type_for_risk = (
        skin_type.capitalize()
    )


    skin_type_encoded = (
        encoder.transform(
            [skin_type_for_risk]
        )[0]
    )


    input_data = pd.DataFrame([{

        "temperature":
            temperature,

        "humidity":
            humidity,

        "uv_index":
            uv_index,

        "wind_speed":
            wind_speed,

        "exposure_hours":
            exposure_hours,

        "sunscreen_used":
            sunscreen_used,

        "moisturizer_used":
            moisturizer_used,

        "skin_type_encoded":
            skin_type_encoded
    }])


    prediction = model.predict(
        input_data
    )[0]


    probabilities_risk = (
        model.predict_proba(
            input_data
        )[0]
    )


    risk_confidence = round(
        max(probabilities_risk) * 100,
        2
    )


    # =====================================================
    # GENERATE PERSONALIZED RECOMMENDATIONS
    # =====================================================

    recommendations = generate_recommendations(

        skin_type=skin_type,

        temperature=temperature,

        humidity=humidity,

        uv_index=uv_index,

        weather_condition=weather_condition
    )


    # =====================================================
    # SAVE COMPLETE ANALYSIS DATA FOR
    # RECOMMENDATION PAGE
    # =====================================================

    session["glow_cast_analysis"] = {

        "city":
            city,

        "skin_type":
            skin_type,

        "skin_confidence":
            skin_confidence,

        "prediction":
            prediction,

        "risk_confidence":
            risk_confidence,

        "temperature":
            temperature,

        "humidity":
            humidity,

        "uv_index":
            uv_index,

        "wind_speed":
            wind_speed,

        "weather_condition":
            weather_condition,

        "exposure_hours":
            exposure_hours,

        "recommendations":
            recommendations
    }


    # =====================================================
    # FINAL TERMINAL OUTPUT
    # =====================================================

    print(
        "\n=============================="
    )

    print(
        "GLOW CAST AI PREDICTION"
    )

    print(
        "=============================="
    )


    print(
        "City:",
        city
    )


    print(
        "Skin Type:",
        skin_type
    )


    print(
        "Skin AI Confidence:",
        skin_confidence
    )


    print(
        "Temperature:",
        temperature,
        "°C"
    )


    print(
        "Humidity:",
        humidity,
        "%"
    )


    print(
        "Weather:",
        weather_condition
    )


    print(
        "UV Index:",
        uv_index
    )


    print(
        "Wind Speed:",
        wind_speed
    )


    print(
        "Risk:",
        prediction
    )


    print(
        "Risk Confidence:",
        risk_confidence,
        "%"
    )


    print(
        "\nGLOW CAST PERSONALIZED EDIT"
    )


    print(
        "Moisturizer:",
        recommendations["moisturizer"]
    )


    print(
        "Sunscreen:",
        recommendations["sunscreen"]
    )


    print(
        "Makeup:",
        recommendations["makeup"]
    )


    print(
        "Daily Ritual:",
        recommendations["daily_ritual"]
    )


    # =====================================================
    # RESULT PAGE
    # =====================================================

    return render_template(

        "result.html",

        city=city,

        skin_type=skin_type,

        skin_confidence=
            skin_confidence,

        prediction=
            prediction,

        confidence=
            risk_confidence,

        temperature=
            temperature,

        humidity=
            humidity,

        uv_index=
            uv_index,

        wind_speed=
            wind_speed,

        weather_condition=
            weather_condition,

        exposure_hours=
            exposure_hours
    )


# =========================================================
# PERSONALIZED RECOMMENDATION PAGE
# =========================================================

@app.route("/recommendation")
def recommendation():

    analysis_data = session.get(
        "glow_cast_analysis"
    )


    # -----------------------------------------------------
    # If recommendation page is opened directly without
    # completing an analysis.
    # -----------------------------------------------------

    if not analysis_data:

        recommendations = generate_recommendations(

            skin_type="normal",

            temperature=30.0,

            humidity=60.0,

            uv_index=5.0,

            weather_condition="Weather unavailable"
        )


        return render_template(

            "recommendation.html",

            city="Unknown Location",

            skin_type="normal",

            skin_confidence=None,

            prediction="Unknown",

            confidence=None,

            temperature=30.0,

            humidity=60.0,

            uv_index=5.0,

            wind_speed=10.0,

            weather_condition="Weather unavailable",

            exposure_hours=None,

            recommendations=recommendations
        )


    # -----------------------------------------------------
    # Get saved recommendation data
    # -----------------------------------------------------

    recommendations = analysis_data.get(
        "recommendations"
    )


    # -----------------------------------------------------
    # Safety fallback if recommendation data is missing
    # -----------------------------------------------------

    if not recommendations:

        recommendations = generate_recommendations(

            skin_type=analysis_data.get(
                "skin_type",
                "normal"
            ),

            temperature=analysis_data.get(
                "temperature",
                30.0
            ),

            humidity=analysis_data.get(
                "humidity",
                60.0
            ),

            uv_index=analysis_data.get(
                "uv_index",
                5.0
            ),

            weather_condition=analysis_data.get(
                "weather_condition",
                "Weather unavailable"
            )
        )


    # -----------------------------------------------------
    # Recommendation Page
    # -----------------------------------------------------

    return render_template(

        "recommendation.html",

        city=analysis_data.get(
            "city",
            "Unknown Location"
        ),

        skin_type=analysis_data.get(
            "skin_type",
            "normal"
        ),

        skin_confidence=analysis_data.get(
            "skin_confidence"
        ),

        prediction=analysis_data.get(
            "prediction",
            "Unknown"
        ),

        confidence=analysis_data.get(
            "risk_confidence"
        ),

        temperature=analysis_data.get(
            "temperature"
        ),

        humidity=analysis_data.get(
            "humidity"
        ),

        uv_index=analysis_data.get(
            "uv_index"
        ),

        wind_speed=analysis_data.get(
            "wind_speed"
        ),

        weather_condition=analysis_data.get(
            "weather_condition",
            "Weather unavailable"
        ),

        exposure_hours=analysis_data.get(
            "exposure_hours"
        ),

        recommendations=recommendations
    )


# =========================================================
# RUN APP
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )

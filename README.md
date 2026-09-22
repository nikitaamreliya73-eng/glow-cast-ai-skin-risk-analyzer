# ✦ GLOW CAST

### AI-Powered Weather-Adaptive Skin Risk Analyzer

GLOW CAST is an AI-powered web application that analyzes a user's skin type from a facial image and combines the result with live environmental conditions to provide a personalized skin-risk assessment and beauty routine.

The system combines **Computer Vision, Machine Learning, Live Weather Data, and personalized recommendations** into one premium web experience.

---

## ✨ Features

* 🧠 **AI Skin Type Detection**

  * Dry
  * Normal
  * Oily

* 📷 **Image-Based Skin Analysis**

  * Upload a facial image
  * CNN-based skin classification

* 🤖 **AI Confidence**

  * Displays the model's confidence for the predicted skin type

* 🌤️ **Live Weather Integration**

  * Temperature
  * Humidity
  * Weather condition
  * Wind speed
  * UV index

* ⚠️ **Environmental Skin Risk**

  * Combines skin type and environmental conditions
  * Provides risk classification and confidence

* 💧 **Personalized Beauty Recommendations**

  * Moisturizer
  * Sunscreen
  * Makeup
  * Daily skincare ritual

* 📊 **AI-Powered Beauty Report**

  * Skin profile
  * Live atmosphere
  * Environmental risk
  * Personalized insight

* 📱 **Responsive Premium UI**

  * Desktop
  * Tablet
  * Mobile

---

## 🧠 Machine Learning

GLOW CAST uses a **MobileNetV2-based Convolutional Neural Network** for skin-type classification.

### Skin Classification

| Class  | Description                |
| ------ | -------------------------- |
| Dry    | Dry skin classification    |
| Normal | Normal skin classification |
| Oily   | Oily skin classification   |

### CNN Architecture

```text
Input Image
     ↓
224 × 224 × 3
     ↓
Data Augmentation
     ↓
MobileNetV2
     ↓
Global Average Pooling
     ↓
Batch Normalization
     ↓
Dense Layer (128)
     ↓
Dropout
     ↓
Softmax
     ↓
Dry / Normal / Oily
```

The final model uses **class-weighted training** to reduce the effect of class imbalance.

---

## 📈 Model Evaluation

The final CNN V6 model was evaluated on a held-out test set.

* **Test Accuracy:** 42.54%
* **Macro F1:** 41.01%
* **Weighted F1:** 43.07%

The V6 experiment reduced the strong Oily-class prediction bias observed in the earlier model version and improved recall for Dry and Normal classes.

> Model confidence represents the model's predicted probability for the selected class and should not be interpreted as diagnostic certainty.

---

## 🌦️ Weather Intelligence

GLOW CAST retrieves live environmental information and uses it as additional context for the skin-risk analysis.

The application considers:

```text
Temperature
Humidity
UV Index
Wind Speed
Weather Condition
        ↓
Environmental Context
        ↓
Skin Risk Assessment
```

This allows the application to connect **skin characteristics with the surrounding environment**.

---

## 💎 Personalized Beauty Intelligence

After analyzing the skin type and environment, GLOW CAST generates a personalized beauty edit.

Example:

```text
Skin Type: Dry
Temperature: 29.9°C
Humidity: 79%

Moisturizer:
Hydrating Barrier Gel-Cream

Sunscreen:
Broad-Spectrum SPF 50 Sunscreen

Makeup:
Hydrating Natural-Finish Foundation

Daily Ritual:
Gentle cleanse → barrier hydration → SPF → natural-finish makeup
```

---

## 🛠️ Tech Stack

### Backend

* Python
* Flask

### Machine Learning

* TensorFlow
* Keras
* Scikit-learn
* MobileNetV2
* NumPy
* Pandas

### Computer Vision

* Pillow

### Data & Model

* Joblib
* OpenWeather API

### Frontend

* HTML5
* CSS3
* JavaScript

### Development

* VS Code
* Git
* GitHub

---

## 📁 Project Structure

```text
glow-cast-ai-skin-risk-analyzer/
│
├── app.py
├── skin_analyzer.py
├── predict_skin.py
├── requirements.txt
├── .gitignore
│
├── app/
│
├── model/
│   ├── best_skin_type_cnn_v6.keras
│   ├── skin_risk_model.pkl
│   └── skin_type_encoder.pkl
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/
│   ├── index.html
│   ├── analyze.html
│   ├── result.html
│   └── recommendation.html
│
└── README.md
```

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/nikitaamreliya73-eng/glow-cast-ai-skin-risk-analyzer.git
```

### 2. Open the project

```bash
cd glow-cast-ai-skin-risk-analyzer
```

### 3. Create a virtual environment

```bash
python -m venv tfenv
```

### 4. Activate the environment

### Windows PowerShell

```powershell
.\tfenv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure environment variables

Create a `.env` file:

```env
OPENWEATHER_API_KEY=your_api_key_here
```

> Never commit your `.env` file or API keys to GitHub.

### 7. Run the application

```bash
python app.py
```

The application will be available at:

```text
http://127.0.0.1:5000
```

---

## 🔄 Application Workflow

```text
User
 │
 ▼
Upload Facial Image
 │
 ▼
CNN Skin Analysis
 │
 ├── Dry
 ├── Normal
 └── Oily
 │
 ▼
AI Confidence
 │
 ▼
Live Weather Data
 │
 ├── Temperature
 ├── Humidity
 ├── UV
 └── Wind
 │
 ▼
Environmental Risk Model
 │
 ▼
AI Beauty Report
 │
 ▼
Personalized Beauty Recommendations
```

---

## 🔐 Security

Sensitive and local development files are excluded from version control.

The repository does not include:

* API keys
* `.env`
* Virtual environments
* Local datasets
* Uploaded user images
* Generated files
* Duplicate quarantine files
* Experimental/old model files

---

## ⚠️ Disclaimer

GLOW CAST is an educational and portfolio project demonstrating machine learning, computer vision, weather integration, and personalized recommendation concepts.

The skin analysis and environmental risk results are **not medical diagnoses** and should not replace professional dermatological advice.

---

## 👩‍💻 Developer

### Nikita Amreliya

Aspiring **Python / AI-ML / Data Science Developer**

🔗 **LinkedIn**
https://www.linkedin.com/in/amreliya-nikita-a63893275

🔗 **GitHub**
https://github.com/nikitaamreliya73-eng

---

## ⭐ Project

**GLOW CAST — AI-Powered Weather-Adaptive Skin Risk Analyzer**

Built with Python, Flask, TensorFlow, Machine Learning, Computer Vision, and Live Weather Intelligence.

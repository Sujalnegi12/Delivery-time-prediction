import pickle
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Delivery Time Predictor",
    page_icon="🍽️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_random_forest_model.pkl"
ENCODER_PATH = BASE_DIR / "label_encoders.pkl"

FEATURE_ORDER = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

# Thresholds (minutes) for the status badge. Tune these to your dataset.
QUICK_LIMIT = 30
NORMAL_LIMIT = 50

VEHICLE_ICONS = {"bike": "🏍️", "scooter": "🛵", "car": "🚗"}
WEATHER_ICONS = {"clear": "☀️", "rainy": "🌧️", "foggy": "🌫️", "snowy": "❄️", "windy": "💨"}
TRAFFIC_ICONS = {"low": "🟢", "medium": "🟡", "high": "🔴"}


# ---------------------------------------------------------
# Zomato-style dark theme
# ---------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito+Sans:wght@400;600;700;800;900&display=swap');

:root {
    --red: #E23744;
    --red-dark: #CB202D;
    --red-soft: #4A2226;
    --bg: #1C1C1C;
    --card: #2A2A2E;
    --card-2: #35353A;
    --ink: #FBEEEA;
    --grey: #B9AEAB;
    --line: #444449;
    --green: #4CC26A;
    --green-soft: #1F3A29;
    --amber: #F5A623;
    --amber-soft: #43331A;
}

html, body, [class*="css"], .stApp, button, input, select {
    font-family: 'Nunito Sans', 'Segoe UI', sans-serif !important;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

.stApp { background: var(--bg); color: var(--ink); }
.block-container { max-width: 760px; padding-top: 0 !important; padding-bottom: 3rem; }

/* Top bar */
.topbar {
    background: linear-gradient(90deg, var(--red-dark), var(--red));
    margin: 0 -9999px 0 -9999px;
    padding: 14px 9999px;
    display: flex;
    align-items: center;
    gap: 10px;
    color: var(--ink);
}
.topbar .logo { font-size: 26px; font-weight: 900; letter-spacing: -0.5px; font-style: italic; }
.topbar .tag { font-size: 13px; font-weight: 600; opacity: .92; margin-left: auto; }

/* Hero */
.hero { padding: 26px 0 8px 0; }
.hero h1 { font-size: 30px; font-weight: 900; color: var(--ink); margin: 0 0 4px 0; line-height: 1.15; }
.hero p { font-size: 15px; color: var(--grey); margin: 0; }

/* Cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--card);
    border: 1px solid var(--line) !important;
    border-radius: 14px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}
.card-title { font-size: 16px; font-weight: 800; color: var(--ink); margin: 0 0 2px 0; }
.card-sub { font-size: 13px; color: var(--grey); margin: 0 0 6px 0; }

/* Labels and inputs */
label p, .stSelectbox label, .stNumberInput label, .stSlider label {
    font-weight: 700 !important;
    color: var(--ink) !important;
    font-size: 14px !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px !important;
    border-color: var(--line) !important;
    background: var(--card-2) !important;
    color: var(--ink) !important;
}
div[data-baseweb="select"] *, div[data-baseweb="input"] input {
    color: var(--ink) !important;
    -webkit-text-fill-color: var(--ink) !important;
}
div[data-baseweb="select"] svg { fill: var(--grey) !important; }
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
    border-color: var(--red) !important;
    box-shadow: 0 0 0 1px var(--red) !important;
}
button[data-testid="stNumberInputStepDown"], button[data-testid="stNumberInputStepUp"] {
    background: var(--card-2) !important;
    color: var(--ink) !important;
}
/* Dropdown menu (rendered in a popover outside the card) */
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] > div {
    background: var(--card-2) !important;
    color: var(--ink) !important;
}
div[data-baseweb="popover"] li { color: var(--ink) !important; }
div[data-baseweb="popover"] li:hover, div[data-baseweb="popover"] li[aria-selected="true"] {
    background: var(--red-soft) !important;
}
/* Slider */
div[data-testid="stSlider"] [role="slider"] { background-color: var(--red) !important; box-shadow: none !important; }
div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div { background: var(--red) !important; }
div[data-testid="stSliderThumbValue"], div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] {
    color: var(--ink) !important;
}

/* Primary button */
div.stButton > button {
    width: 100%;
    height: 54px;
    font-size: 17px;
    font-weight: 800;
    border-radius: 12px;
    border: none;
    color: var(--ink);
    background: var(--red);
    transition: background .15s ease, transform .05s ease;
}
div.stButton > button:hover { background: var(--red-dark); color: var(--ink); border: none; }
div.stButton > button:active { transform: scale(.99); }
div.stButton > button:focus-visible { outline: 3px solid var(--ink); outline-offset: 2px; }

/* ETA card */
.eta {
    background: var(--card);
    border: 1px solid var(--line);
    border-left: 6px solid var(--red);
    border-radius: 14px;
    padding: 22px 24px;
    margin-top: 18px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}
.eta-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
.eta-label { font-size: 14px; color: var(--grey); font-weight: 600; }
.eta-time { font-size: 56px; font-weight: 900; color: var(--ink); line-height: 1; margin-top: 4px; }
.eta-time span { font-size: 22px; font-weight: 800; color: var(--grey); margin-left: 6px; }
.eta-arrive { font-size: 15px; color: var(--ink); margin-top: 8px; font-weight: 600; }
.badge { padding: 6px 12px; border-radius: 999px; font-size: 13px; font-weight: 800; }
.badge.quick { background: var(--green-soft); color: var(--green); }
.badge.normal { background: var(--amber-soft); color: var(--amber); }
.badge.slow { background: var(--red-soft); color: #FF7A85; }

/* Tracker steps */
.steps { display: flex; margin-top: 20px; }
.step { flex: 1; text-align: center; position: relative; font-size: 12px; color: var(--grey); font-weight: 700; }
.step .dot {
    width: 14px; height: 14px; border-radius: 50%;
    background: var(--red); margin: 0 auto 6px auto; position: relative; z-index: 2;
}
.step:not(:last-child)::after {
    content: ""; position: absolute; top: 6px; left: 50%; width: 100%;
    height: 2px; background: var(--red); z-index: 1;
}
.step .mins { display: block; color: var(--ink); font-size: 13px; font-weight: 800; }

/* Summary chips */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px; }
.chip {
    background: var(--card-2); color: var(--ink); border-radius: 10px;
    padding: 8px 12px; font-size: 13px; font-weight: 700;
}
.chip small { display: block; color: var(--grey); font-weight: 600; font-size: 11px; }

[data-testid="stCaptionContainer"], .stCaption { color: var(--grey) !important; }
.foot { text-align: center; color: var(--grey); font-size: 12px; margin-top: 28px; }

@media (max-width: 600px) {
    .hero h1 { font-size: 24px; }
    .eta-time { font-size: 44px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load model and encoders
# ---------------------------------------------------------
@st.cache_resource
def load_pickle(path: Path):
    with open(path, "rb") as file:
        return pickle.load(file)


try:
    model = load_pickle(MODEL_PATH)
    label_encoders = load_pickle(ENCODER_PATH)
except FileNotFoundError as e:
    st.error(
        f"Required file not found: {e.filename}\n\n"
        "Keep these files in the same folder as main.py:\n"
        "- best_random_forest_model.pkl\n"
        "- label_encoders.pkl"
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the model or encoders: {e}")
    st.stop()


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def with_icon(value: str, icons: dict) -> str:
    return f"{icons.get(str(value).lower(), '•')} {value}"


def encode_value(column: str, value: str) -> int:
    return int(label_encoders[column].transform([value])[0])


def predict_minutes(inputs: dict) -> float:
    row = {
        "Distance_km": inputs["distance"],
        "Weather": encode_value("Weather", inputs["weather"]),
        "Traffic_Level": encode_value("Traffic_Level", inputs["traffic"]),
        "Time_of_Day": encode_value("Time_of_Day", inputs["time_of_day"]),
        "Vehicle_Type": encode_value("Vehicle_Type", inputs["vehicle"]),
        "Preparation_Time_min": inputs["prep"],
        "Courier_Experience_yrs": inputs["experience"],
    }
    # Use the exact column order the model was trained with, if it stored it.
    columns = list(getattr(model, "feature_names_in_", FEATURE_ORDER))
    frame = pd.DataFrame([row])[columns]
    return max(0.0, float(model.predict(frame)[0]))


def status_badge(minutes: float):
    if minutes <= QUICK_LIMIT:
        return "quick", "⚡ Quick delivery"
    if minutes <= NORMAL_LIMIT:
        return "normal", "👍 On time"
    return "slow", "⏳ May take longer"


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    """
<div class="topbar">
    <div class="logo">🍽️ DeliveryPredict</div>
    <div class="tag">Random Forest ETA</div>
</div>
<div class="hero">
    <h1>How long will your food take?</h1>
    <p>Enter the delivery details and get an estimated delivery time.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Inputs
# ---------------------------------------------------------
with st.container(border=True):
    st.markdown(
        '<p class="card-title">📍 Trip details</p>'
        '<p class="card-sub">Where and when the order is travelling</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        distance = st.slider(
            "Distance (km)", min_value=0.59, max_value=19.99, value=7.50, step=0.01
        )
        time_of_day = st.selectbox(
            "Time of day", list(label_encoders["Time_of_Day"].classes_)
        )
    with c2:
        weather = st.selectbox(
            "Weather",
            list(label_encoders["Weather"].classes_),
            format_func=lambda v: with_icon(v, WEATHER_ICONS),
        )
        traffic = st.selectbox(
            "Traffic level",
            list(label_encoders["Traffic_Level"].classes_),
            format_func=lambda v: with_icon(v, TRAFFIC_ICONS),
        )

with st.container(border=True):
    st.markdown(
        '<p class="card-title">🧑‍🍳 Restaurant and delivery partner</p>'
        '<p class="card-sub">Kitchen speed and rider details</p>',
        unsafe_allow_html=True,
    )
    c3, c4, c5 = st.columns(3)
    with c3:
        prep = st.number_input(
            "Preparation time (min)", min_value=5, max_value=29, value=15, step=1
        )
    with c4:
        experience = st.number_input(
            "Rider experience (yrs)", min_value=0.0, max_value=9.0, value=2.0, step=0.5
        )
    with c5:
        vehicle = st.selectbox(
            "Vehicle",
            list(label_encoders["Vehicle_Type"].classes_),
            format_func=lambda v: with_icon(v, VEHICLE_ICONS),
        )

predict_clicked = st.button("Predict delivery time", type="primary")

# ---------------------------------------------------------
# Result
# ---------------------------------------------------------
if predict_clicked:
    try:
        minutes = predict_minutes(
            {
                "distance": distance,
                "weather": weather,
                "traffic": traffic,
                "time_of_day": time_of_day,
                "vehicle": vehicle,
                "prep": prep,
                "experience": experience,
            }
        )
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    minutes_rounded = round(minutes)
    arrival = datetime.now(ZoneInfo("Asia/Kolkata")) + timedelta(minutes=minutes)
    badge_class, badge_text = status_badge(minutes)

    # Split the total time into the kitchen part and the on-road part
    prep_part = min(prep, minutes_rounded)
    road_part = max(minutes_rounded - prep_part, 0)

    st.markdown(
        f"""
<div class="eta">
<div class="eta-row">
<div>
<div class="eta-label">Estimated delivery time</div>
<div class="eta-time">{minutes_rounded}<span>mins</span></div>
</div>
<div class="badge {badge_class}">{badge_text}</div>
</div>
<div class="eta-arrive">Arriving around <b>{arrival.strftime('%I:%M %p').lstrip('0')}</b></div>
<div class="steps">
<div class="step"><div class="dot"></div>Order placed<span class="mins">0 min</span></div>
<div class="step"><div class="dot"></div>Food ready<span class="mins">{prep_part} min</span></div>
<div class="step"><div class="dot"></div>Delivered<span class="mins">{minutes_rounded} min</span></div>
</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.write("")
    with st.container(border=True):
        st.markdown('<p class="card-title">Order summary</p>', unsafe_allow_html=True)
        chips = [
            ("Distance", f"{distance:.2f} km"),
            ("Weather", with_icon(weather, WEATHER_ICONS)),
            ("Traffic", with_icon(traffic, TRAFFIC_ICONS)),
            ("Time of day", time_of_day),
            ("Vehicle", with_icon(vehicle, VEHICLE_ICONS)),
            ("Preparation", f"{prep} min"),
            ("Rider experience", f"{experience:.1f} yrs"),
        ]
        chip_html = "".join(
            f'<div class="chip"><small>{label}</small>{value}</div>' for label, value in chips
        )
        st.markdown(f'<div class="chips">{chip_html}</div>', unsafe_allow_html=True)
        st.caption(
            f"About {prep_part} min in the kitchen and {road_part} min on the road (approximate split)."
        )

st.markdown(
    '<div class="foot">Food Delivery Time Prediction • Random Forest Regression</div>',
    unsafe_allow_html=True,
)

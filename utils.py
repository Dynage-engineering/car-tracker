# filepath: /Users/mac/Desktop/devprojects/playground/car-tracker/utils.py
import gspread
from google.oauth2.service_account import Credentials
import datetime
import pandas as pd
import streamlit as st  # needed to access st.secrets
from ultralytics import YOLO  # ensure ultralytics is installed
import av


model = YOLO("yolov8n.pt")


def get_google_sheet_worksheet():
    SCOPE = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive",
    ]
    # Load credentials from Streamlit secrets
    credentials = Credentials.from_service_account_info(st.secrets["gcp"], scopes=SCOPE)
    gc = gspread.authorize(credentials)
    sh = gc.open("VehicleData")
    worksheet = sh.sheet1
    return worksheet


def store_today_data(car_count, bus_count, truck_count):
    worksheet = get_google_sheet_worksheet()
    today = datetime.date.today().strftime("%Y-%m-%d")
    worksheet.append_row([today, car_count, truck_count, bus_count])


DAY = 60 * 60 * 24  # seconds in a day


@st.cache_data(ttl=DAY)
def load_sheet_data():
    worksheet = get_google_sheet_worksheet()
    records = worksheet.get_all_records()
    if not records:
        return pd.DataFrame(
            columns=["Date", "Number of Cars", "Number of Truck", "Number of Bus"]
        )
    df = pd.DataFrame(records)
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values("Date", inplace=True)
    return df


def transform(frame):
    # Convert incoming frame to ndarray (BGR)
    img = frame.to_ndarray(format="bgr24")
    results = model(img)
    if results and results[0].boxes is not None:
        # Update the global counts by reading and updating st.session_state
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        for cls in classes:
            label = model.model.names[cls]
            if label in st.session_state.global_counts:
                # Direct update (the WebRTC callback works asynchronously)
                st.session_state.global_counts[label] += 1
        # Annotate frame (draw boxes, etc.)
        annotated = results[0].plot()
    else:
        annotated = img

    # Return new frame in correct format
    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

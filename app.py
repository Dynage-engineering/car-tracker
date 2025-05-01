import streamlit as st
import numpy as np
import pandas as pd
import time
import datetime
from ultralytics import YOLO  # ensure ultralytics is installed
from utils import store_today_data, load_sheet_data, model, transform
from streamlit_webrtc import webrtc_streamer
import av
from streamlit_autorefresh import st_autorefresh

# Initialize session_state safely
st.session_state.setdefault("global_counts", {"car": 0, "bus": 0, "truck": 0})

# --- Load YOLOv8 model ---

st.header("Vehicle Detection App")
st.subheader("Detect vehicles in a video stream")
st.markdown(
    """
    This app uses YOLOv8 and DeepSORT for vehicle detection and tracking.
    - **YOLOv8**: A state-of-the-art object detection model.
    - **DeepSORT**: A tracking algorithm that associates detected objects across frames.
    """
)

# --- UI Columns ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Detected Vehicles")
    # Load historical data from Google Sheets and plot bar chart
    sheet_df = load_sheet_data()
    if sheet_df.empty:
        st.write("No historical data available yet.")
    else:
        st.bar_chart(sheet_df.set_index("Date"), use_container_width=True)
        st.write("Detected vehicles over time.")

    # Live metrics display refreshed automatically
    mcols = st.columns(3)
    mcols[0].metric("Car", st.session_state.global_counts["car"])
    mcols[1].metric("Bus", st.session_state.global_counts["bus"])
    mcols[2].metric("Truck", st.session_state.global_counts["truck"])

with col2:
    st.subheader("Camera Stream")
    st.write("Camera stream will be displayed here using streamlit_webrtc.")
    webrtc_streamer(
        key="example",
        video_frame_callback=lambda frame: transform(frame),
        rtc_configuration={
            "max_frames_per_second": 30,
            "iceServers": [
                {
                    "urls": [
                        "stun:stun.l.google.com:19302",
                        "stun:stun1.l.google.com:19302",
                        "stun:stun2.l.google.com:19302",
                    ]
                }
            ],
        },
    )


# Auto-refresh the app every second without using threading for UI updates
st_autorefresh(interval=60000, key="refresh_metrics")

# Optionally, store end-of-day data
now = datetime.datetime.now().time()
if now.hour == 23 and now.minute >= 59:
    store_today_data(
        car_count=st.session_state.global_counts["car"],
        bus_count=st.session_state.global_counts["bus"],
        truck_count=st.session_state.global_counts["truck"],
    )
    st.success("Today's data has been automatically stored to Google Sheets.")

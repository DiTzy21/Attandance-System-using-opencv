# show_attendance.py

import streamlit as st
import pandas as pd
import os
import time
from datetime import datetime

def main():
    st.title("Attendance List")

    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

    if os.path.exists(f"Attendance/Attendance_{date}.csv"):
        df = pd.read_csv(f"Attendance/Attendance_{date}.csv")
        st.dataframe(df.style.highlight_max(axis=0))
    else:
        st.write("No attendance records for today.")

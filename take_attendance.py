import cv2
import pickle
import numpy as np
import os
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import datetime
import csv

def main():
    st.title("Take Attendance")

   
    if 'take_attendance_now' not in st.session_state:
        st.session_state.take_attendance_now = False

    if 'video' not in st.session_state:
        st.session_state.video = None

    if 'take_attendance' not in st.session_state:
        st.session_state.take_attendance = False

    if st.button("Start Webcam"):
        st.session_state.video = cv2.VideoCapture(0)
        st.session_state.take_attendance_now = True

    if st.session_state.take_attendance_now:
        facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

        with open('data/users.pkl', 'rb') as w:
            USERS = pickle.load(w)
        with open('data/faces_data.pkl', 'rb') as f:
            FACES = pickle.load(f)

        names = [user['name'] for user in USERS]

        if len(FACES) != len(names):
            st.error("Mismatch between number of faces and user information entries. Please check your data.")
            return

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(FACES, names)

        FRAME_WINDOW = st.image([])
        COL_NAMES = ['NAME', 'ID', 'MAJOR', 'TIME']

        stop_button = st.button("Stop Webcam")
        take_attendance_button = st.button("Take Attendance Now")

        if take_attendance_button:
            st.session_state.take_attendance = True

        if stop_button:
            st.session_state.video.release()
            st.session_state.take_attendance_now = False
            st.stop()

        while st.session_state.take_attendance_now:
            ret, frame = st.session_state.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                output_name = knn.predict(resized_img)[0]

                user_info = next(user for user in USERS if user['name'] == output_name)
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
                exist = os.path.isfile(f"Attendance/Attendance_{date}.csv")
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y-80), (x+w, y), (50, 50, 255), -1)  # Adjust height for additional text
                cv2.putText(frame, f"Name: {user_info['name']}", (x, y-60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"SID: {user_info['id']}", (x, y-40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Major: {user_info['major']}", (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                attendance = [str(user_info['name']), str(user_info['id']), str(user_info['major']), str(timestamp)]

                if st.session_state.take_attendance:
                    st.success("Attendance Taken..")
                    if exist:
                        with open(f"Attendance/Attendance_{date}.csv", "+a") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(attendance)
                        csvfile.close()
                    else:
                        with open(f"Attendance/Attendance_{date}.csv", "+a") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(COL_NAMES)
                            writer.writerow(attendance)
                        csvfile.close()
                    st.session_state.take_attendance = False

            FRAME_WINDOW.image(frame, channels='BGR')

            # Remove cv2.waitKey and use Streamlit's stop button to break the loop
            if stop_button:
                break

if __name__ == '__main__':
    main()
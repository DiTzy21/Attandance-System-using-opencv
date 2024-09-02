import cv2
import pickle
import numpy as np
import os
import streamlit as st
from PIL import Image

def main():
    st.title("Add New Face (BETA)")

    name = st.text_input("Enter Your Name:")
    user_id = st.text_input("Enter Your ID:")
    major = st.text_input("Enter Your Major:")
    add_face_button = st.button("Add Face")

    if add_face_button and name and user_id and major:
        video = cv2.VideoCapture(0)
        facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

        faces_data = []
        i = 0

        FRAME_WINDOW = st.image([])

        while len(faces_data) < 100:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
                resized_img = cv2.resize(crop_img, (50, 50))
                if i % 10 == 0:
                    faces_data.append(resized_img)
                i += 1
                cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

            FRAME_WINDOW.image(frame, channels='BGR')
            if len(faces_data) >= 100:
                break

        video.release()

        faces_data = np.asarray(faces_data).reshape(100, -1)

        user_info = {'name': name, 'id': user_id, 'major': major}

        if 'users.pkl' not in os.listdir('data/'):
            users = [user_info]*100
            with open('data/users.pkl', 'wb') as f:
                pickle.dump(users, f)
        else:
            with open('data/users.pkl', 'rb') as f:
                users = pickle.load(f)
            users.extend([user_info]*100)
            with open('data/users.pkl', 'wb') as f:
                pickle.dump(users, f)

        if 'faces_data.pkl' not in os.listdir('data/'):
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces_data, f)
        else:
            with open('data/faces_data.pkl', 'rb') as f:
                faces = pickle.load(f)
            faces = np.append(faces, faces_data, axis=0)
            with open('data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces, f)

        st.success(f"Faces for {name} added successfully!")

if __name__ == '__main__':
    main()
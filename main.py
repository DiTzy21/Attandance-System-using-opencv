# main.py

import streamlit as st


st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Add New Face (BETA)", "Take Attendance", "Show Attendance"])

if option == "Home":
    st.title("Face Recognition Attendance System")
    st.write("Welcome to the Face Recognition Attendance System. Use the sidebar to navigate. We are excited to introduce you to our innovative solution designed to streamline and enhance the attendance tracking process")
elif option == "Add New Face (BETA)":
    import add_faces
    add_faces.main()

elif option == "Take Attendance":
    import take_attendance
    take_attendance.main()

elif option == "Show Attendance":
    import show_attendance
    show_attendance.main()
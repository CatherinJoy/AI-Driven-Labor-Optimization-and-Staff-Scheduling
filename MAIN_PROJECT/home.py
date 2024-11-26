import streamlit as st
from PIL import Image
def home():
    st.title("Home")
    st.write("Welcome to the scheduling application. Please select an option from the sidebar.")

     # Displaying the image in full screen
    images = Image.open("chicking.png")
    st.image(images)
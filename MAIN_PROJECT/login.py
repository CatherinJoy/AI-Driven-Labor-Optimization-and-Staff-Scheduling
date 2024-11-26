import streamlit as st
from sales_forecast import sales_forecast_page
from PIL import Image
from streamlit_option_menu import option_menu
from home import home
from automated_staff_scheduling import automated_staff_scheduling_page
from employee_management import employee_management_page
from users import get_user_info, change_password, load_passwords
from settings import settings_page
import json
from logged_in_state import load_state, save_state

# Function to authenticate user
@st.cache_data
def authenticate_user(username, password):
    passwords = load_passwords()
    if username in passwords and passwords[username] == password:
        return True
    return False

def login_page():
    state = load_state()
    if state["logged_in"]:
        # User is already logged in, show a message
        st.write(f"You are already logged in as {state['username']}.")
    else:
        st.title("Welcome!")
        st.write("Please login to access the application.")
        # Load and display image
        image = Image.open("Schedule1.jpg")
        st.image(image, width=400)
        # User input fields
        entered_username = st.text_input("Username")
        entered_password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate_user(entered_username, entered_password):
                state["logged_in"] = True
                state["username"] = entered_username
                save_state(state)
            else:
                st.error("Invalid username or password. Please try again.")

def main():
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    state = load_state()
    if state["logged_in"]:
        # User is logged in, update session state
        st.session_state.logged_in = state["logged_in"]
        st.session_state.username = state["username"]

        # User is logged in, display the selected page
        with st.sidebar:
            selected_page = option_menu(
                menu_title="Menu",
                options=["Home", "Sales Forecast", "Automated Staff Scheduling", "Employee Management", "Settings", "Logout"],
                icons=["house-fill", "graph-up-arrow", "pie-chart-fill", "people-fill", "gear-fill", "box-arrow-right"],
                menu_icon="list",
                default_index=0,
            )
        if selected_page == "Home":
            home()
        elif selected_page == "Sales Forecast":
            sales_forecast_page()
        elif selected_page == "Automated Staff Scheduling":
            automated_staff_scheduling_page()
        elif selected_page == "Employee Management":
            employee_management_page()
        elif selected_page == "Settings":
            settings_page()
        elif selected_page == "Logout":
            state["logged_in"] = False
            state["username"] = None
            save_state(state)
    else:
        login_page()
if __name__ == "__main__":
    main()
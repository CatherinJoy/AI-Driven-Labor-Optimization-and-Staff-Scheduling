import streamlit as st
from users import authorized_users, get_user_info, change_password

def settings_page():
    st.title("Settings")
    logged_in_user = st.session_state.username

    if logged_in_user:
        user_data = get_user_info(logged_in_user)  # Fetch user data using get_user_info function
        if user_data:
            st.write(f"Name: {user_data.get('name', 'Unknown')}")
            st.write(f"Email: {user_data.get('email', 'Unknown')}")
            st.write(f"Contact: {user_data.get('contact', 'Unknown')}")

            st.subheader("Change Password")
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")

            if st.button("Change Password"):
                # Call the change_password function to update the password
                change_result = change_password(logged_in_user, current_password, new_password, confirm_password)
                if change_result == "success":
                    st.success("Password changed successfully.")
                elif change_result == "incorrect":
                    st.error("Incorrect current password. Please try again.")
                else:
                    st.error("New password and confirm password do not match.")
        else:
            st.error("User data not found.")
    else:
        st.error("You need to login first to view the settings.")
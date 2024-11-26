import streamlit as st
import json

# Load passwords from passwords.json file
def load_passwords():
    try:
        with open("passwords.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Return an empty dictionary if the file doesn't exist
        return {}

# Save passwords to passwords.json file
def save_passwords(authorized_users):
    with open("passwords.json", "w") as f:
        json.dump(authorized_users, f)

# Load passwords from file
authorized_users = load_passwords()

def change_password(username, current_password, new_password, confirm_password):
    authorized_users = load_passwords()
    current_user_password = authorized_users.get(username)

    # Validate current password
    if current_user_password != current_password:
        return "Incorrect current password. Please try again."

    # Validate new password and confirmation
    if new_password != confirm_password:
        return "New password and confirm password do not match."

    # Update password in authorized_users dictionary
    authorized_users[username] = new_password

    # Save passwords to file
    save_passwords(authorized_users)

    return "success"


def get_user_info(username):
    # This is a placeholder function
    # In a real application, you might retrieve user information from a database or some other source
    user_info = {
        "user1": {"name": "John Doe", "email": "john@example.com", "contact": "1234567890"},
        "user2": {"name": "Jane Smith", "email": "jane@example.com", "contact": "9876543210"},
        #"user3": {"name": "John Doe", "email": "john@example.com", "contact": "1234567890"},
        #"user4": {"name": "Jane Smith", "email": "jane@example.com", "contact": "9876543210"},
        # Add more users as needed
    }
    return user_info.get(username, {})
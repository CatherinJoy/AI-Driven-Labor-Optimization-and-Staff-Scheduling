# employee_management.py
import streamlit as st
import pandas as pd

# Function to read emp_details.csv file and return DataFrame
def read_csv():
    df = pd.read_csv(r'C:\Users\Carolin\Desktop\\employee_details_data.csv')
    if not {'emp_id', 'role', 'skills', 'availability'}.issubset(df.columns):
        raise ValueError("CSV file is missing required columns")
    return df

# Function to display DataFrame
def display_dataframe(df):
    st.write(df)

# Function to write emp_details.csv file
def write_csv(df):
    df.to_csv(r'C:\Users\Carolin\Desktop\\employee_details_data.csv', index=False)

# Function to add new employee details
def add_employee(df, emp_id, role, skills, availability):
    if df.empty:
        df = pd.DataFrame(columns=['emp_id', 'role', 'skills', 'availability'])
    new_row = {'emp_id': emp_id, 'role': role, 'skills': skills, 'availability': availability}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df

# Function to delete employee details
def delete_employee(df, emp_id):
    df = df[~(df['emp_id'] == emp_id)]
    return df

# Function to edit employee details
def edit_employee(df, old_emp_id, new_emp_id, field, new_value):
    # Check if the DataFrame contains the specified old_emp_id
    if (df['emp_id'] == old_emp_id).any():
        idx = df.loc[df['emp_id'] == old_emp_id].index[0]  # Get the index of the first occurrence
        if field == 'emp_id':
            df.at[idx, field] = new_emp_id  # Update the employee ID
        else:
            df.at[idx, field] = new_value  # Update the other fields
        return df
    else:
        st.warning(f"Employee with ID {old_emp_id} not found.")
        return df

# Main function for employee management page
def employee_management_page():
    st.title("Employee Management")

    # Employee Directory section
    st.header("Employee Directory")
    df = read_csv()
    display_dataframe(df)

    # Manage Employees section
    st.header("Manage Employees")
    option = st.selectbox('Select an option', ['Add', 'Delete', 'Edit'])

    if option == 'Add':
        emp_id = st.text_input('Employee ID')
        role = st.text_input('Role')
        skills = st.text_input('Skills')
        availability = st.text_input('Availability')
        if st.button('Add Employee'):
            df = add_employee(df, emp_id, role, skills, availability)
            st.success('Employee added successfully!')
            write_csv(df)  # Save changes to CSV file

    elif option == 'Delete':
        emp_id_to_delete = st.text_input('Enter Employee ID to delete')
        if emp_id_to_delete:
            try:
                emp_id_to_delete = int(emp_id_to_delete)  # Convert input to integer
            except ValueError:
                st.warning('Invalid Employee ID. Please enter a valid integer.')
                return
            if st.button('Delete Employee'):
                df = delete_employee(df, emp_id_to_delete)
                st.success('Employee deleted successfully!')
                write_csv(df)  # Save changes to CSV file

    elif option == 'Edit':
        emp_id_to_edit = st.text_input('Enter Employee ID to edit')
        if emp_id_to_edit:
            try:
                emp_id_to_edit = int(emp_id_to_edit)  # Convert input to integer
            except ValueError:
                st.warning('Invalid Employee ID. Please enter a valid integer.')
                return
            field_to_edit = st.selectbox('Select field to edit', ['emp_id', 'role', 'skills', 'availability'])
            new_value = st.text_input(f'Enter new value for {field_to_edit}')
            if field_to_edit == 'emp_id':
                try:
                    new_value = int(new_value)  # Convert input to integer for employee ID
                except ValueError:
                    st.warning('Invalid Employee ID. Please enter a valid integer.')
                    return
            if st.button('Edit Employee'):
                df = edit_employee(df, emp_id_to_edit, new_value, field_to_edit, new_value)
                st.success('Employee details updated successfully!')
                write_csv(df)  # Save changes to CSV file
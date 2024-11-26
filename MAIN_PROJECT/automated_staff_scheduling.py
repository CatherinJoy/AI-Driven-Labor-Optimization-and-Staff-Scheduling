import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import csv
from collections import defaultdict
from datetime import datetime, timedelta
import random

# Updated dictionaries including manager and assistant manager
average_labor_hours_per_unit = {
    'Manager': 0.0001618,
    'Assistant Manager': 0.0001618,
    'Cook': 0.0008269,
    'Cashier': 0.0003236,
    'Server': 0.0004853,
}

role_specific_factors = {
    'Manager': 0.58,
    'Assistant Manager': 0.58,
    'Cook': 0.59,
    'Cashier': 0.595,
    'Server': 0.6,
}

day_of_week_factors = {
    'Monday': 1.780376,
    'Tuesday': 1.772589,
    'Wednesday': 1.774399,
    'Thursday': 1.807055,
    'Friday': 1.851661,
    'Saturday': 2,
    'Sunday': 1.913906,
}

def load_csv(file_path):
    data = []
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    return data

def prepare_employee_data(employee_details):
    employees = {}
    for emp in employee_details:
        emp_id = emp['emp_id']
        primary_role = emp['role']
        skills = emp['skills'].split(',')
        availability = parse_availability(emp['availability'])
        employees[emp_id] = {'primary_role': primary_role, 'skills': skills, 'availability': availability, 'assigned_shifts': []}
    return employees

def prepare_role_skills(role_skills):
    roles = defaultdict(list)
    for role_skill in role_skills:
        role = role_skill['role']
        skills = role_skill['skills'].split(',')
        roles[role] = skills
    return roles

def parse_time(time_str):
    return datetime.strptime(time_str, '%I %p')

def parse_availability(avail_str):
    return [parse_time(slot.strip()) for slot in avail_str.split(',')]

# Define shift timings
shift_timings = {'morning': (10, 14), 'afternoon': (14, 23)}

# Function to check employee availability for a shift
def is_available(employee_availability, shift_start, shift_end):
    for slot in employee_availability:
        print(f"Checking slot {slot} for shift from {shift_start} to {shift_end}")
        if shift_start.time() <= slot.time() <= shift_end.time():
            print("Employee available for shift.")
            return True
    print("Employee not available for shift.")
    return False

# Function to check if a shift combination is valid
def is_valid_shift_combination(assigned_shifts, shift_start, shift_end):
    morning_shift = (datetime.strptime('10 AM', '%I %p'), datetime.strptime('2 PM', '%I %p'))
    afternoon_shift = (datetime.strptime('2 PM', '%I %p'), datetime.strptime('11 PM', '%I %p'))
    evening_shift = (datetime.strptime('6 PM', '%I %p'), datetime.strptime('11 PM', '%I %p'))

    new_shift = (shift_start.time(), shift_end.time())

    if new_shift == morning_shift and afternoon_shift in [shift for shift in assigned_shifts]:
        return False
    if new_shift == afternoon_shift and (morning_shift in [shift for shift in assigned_shifts] or evening_shift in [shift for shift in assigned_shifts]):
        return False
    if new_shift == evening_shift and afternoon_shift in [shift for shift in assigned_shifts]:
        return False

    return True

def assign_employee_to_shift(day, role, shift_start, shift_end, available_employees, assigned_employees, shift_demand):
    print(f"Assigning employees to {role} shift on {day} from {shift_start} to {shift_end}")

    # Check if the shift has already been assigned
    if (day, shift_start, shift_end, role) in assigned_employees.get(day, []):
        print(f"{role} shift on {day} from {shift_start} to {shift_end} already assigned.")
        return []

    role_available_employees = {emp_id: emp_info for emp_id, emp_info in available_employees.items() if emp_info['primary_role'] == role}

    if not role_available_employees:
        print(f"No available employees with primary role '{role}' for {day} {shift_start}-{shift_end}")
        return []

    # Shuffle the available employees to ensure different schedules
    available_employee_ids = list(role_available_employees.keys())
    random.shuffle(available_employee_ids)

    likelihoods = {}
    for emp_id in available_employee_ids:
        emp_info = role_available_employees[emp_id]
        # Check if the employee is already assigned to a shift on this day
        if emp_id in assigned_employees.get(day, []):
            likelihoods[emp_id] = 0  # Skip this employee
            continue

        # Check if the employee is already assigned to a shift during the same time
        overlapping_shifts = [shift for shift in emp_info['assigned_shifts'] if shift[0] <= shift_start <= shift[1] or shift[0] <= shift_end <= shift[1]]
        if overlapping_shifts:
            likelihoods[emp_id] = 0  # Skip this employee
            continue

        # Check if the new shift combination is valid
        if not is_valid_shift_combination(emp_info['assigned_shifts'], shift_start, shift_end):
            likelihoods[emp_id] = 0  # Skip this employee
            continue

        likelihood = is_available(emp_info['availability'], shift_start, shift_end)
        likelihoods[emp_id] = likelihood
        print(f"Likelihood of employee {emp_id} for {role} shift: {likelihood}")

    # Choose available employees with highest likelihood
    sorted_likelihoods = sorted(likelihoods.items(), key=lambda x: x[1], reverse=True)
    assigned_employees = []
    for emp_id, likelihood in sorted_likelihoods:
        if likelihood:
            assigned_employees.append(emp_id)
            print(f"Assigned employee {emp_id} to {role} shift")
            if len(assigned_employees) >= shift_demand:
                break

    return assigned_employees

# Define function to generate schedule
def generate_schedule(predicted_demand, employees, shift_timings):
    schedule = []
    assigned_employees = {}  # Dictionary to keep track of assigned employees

    for entry in predicted_demand:
        date = entry['date']
        labor_demand = {role: int(float(entry[role])) for role in entry if role != 'date'}  # Convert to integer

        # Assign fixed shifts for manager and assistant_manager
        manager_shift_start_10_2 = datetime.strptime(f"{date} 10:00", "%Y-%m-%d %H:%M")
        manager_shift_end_10_2 = datetime.strptime(f"{date} 14:00", "%Y-%m-%d %H:%M")
        manager_shift_start_6_11 = datetime.strptime(f"{date} 18:00", "%Y-%m-%d %H:%M")
        manager_shift_end_6_11 = datetime.strptime(f"{date} 23:00", "%Y-%m-%d %H:%M")
        assistant_manager_shift_start = datetime.strptime(f"{date} 14:00", "%Y-%m-%d %H:%M")
        assistant_manager_shift_end = datetime.strptime(f"{date} 23:00", "%Y-%m-%d %H:%M")

        manager_available_employees = {emp_id: emp_info for emp_id, emp_info in employees.items() if emp_info['primary_role'] == 'manager'}
        assistant_manager_available_employees = {emp_id: emp_info for emp_id, emp_info in employees.items() if emp_info['primary_role'] == 'assistant_manager'}

        if manager_available_employees:
            manager_emp_id = next(iter(manager_available_employees))
            schedule.append({'date': date, 'emp_id': manager_emp_id, 'shift_start': manager_shift_start_10_2, 'shift_end': manager_shift_end_10_2, 'role': 'manager'})
            schedule.append({'date': date, 'emp_id': manager_emp_id, 'shift_start': manager_shift_start_6_11, 'shift_end': manager_shift_end_6_11, 'role': 'manager'})
            assigned_employees.setdefault(date, []).append(manager_emp_id)

        if assistant_manager_available_employees:
            assistant_manager_emp_id = next(iter(assistant_manager_available_employees))
            schedule.append({'date': date, 'emp_id': assistant_manager_emp_id, 'shift_start': assistant_manager_shift_start, 'shift_end': assistant_manager_shift_end, 'role': 'assistant_manager'})
            assigned_employees.setdefault(date, []).append(assistant_manager_emp_id)

        for role, demand in labor_demand.items():
            if role not in ['manager', 'assistant_manager']:
                print(f"Demand for {role} on {date}: {demand}")

                # Assign employees for morning shift
                morning_shift_start = datetime.strptime(f"{date} 10:00", "%Y-%m-%d %H:%M")
                morning_shift_end = datetime.strptime(f"{date} 14:00", "%Y-%m-%d %H:%M")
                morning_shift_demand = 1  # At least one employee per shift
                print(f"Morning shift demand for {role} on {date}: {morning_shift_demand}")

                morning_assigned_employees = assign_employee_to_shift(date, role, morning_shift_start, morning_shift_end, employees, assigned_employees, morning_shift_demand)
                for emp_id in morning_assigned_employees:
                    schedule.append({'date': date, 'emp_id': emp_id, 'shift_start': morning_shift_start, 'shift_end': morning_shift_end, 'role': role})
                    assigned_employees.setdefault(date, []).append(emp_id)

                    # If an employee is assigned the morning shift, also assign the evening shift
                    evening_shift_start = datetime.strptime(f"{date} 18:00", "%Y-%m-%d %H:%M")
                    evening_shift_end = datetime.strptime(f"{date} 23:00", "%Y-%m-%d %H:%M")

                    schedule.append({'date': date, 'emp_id': emp_id, 'shift_start': evening_shift_start, 'shift_end': evening_shift_end, 'role': role})
                    assigned_employees.setdefault(date, []).append(emp_id)

                # Assign employees for afternoon shift
                afternoon_shift_start = datetime.strptime(f"{date} 14:00", "%Y-%m-%d %H:%M")
                afternoon_shift_end = datetime.strptime(f"{date} 23:00", "%Y-%m-%d %H:%M")
                afternoon_shift_demand = demand - morning_shift_demand
                print(f"Afternoon shift demand for {role} on {date}: {afternoon_shift_demand}")

                afternoon_assigned_employees = assign_employee_to_shift(date, role, afternoon_shift_start, afternoon_shift_end, employees, assigned_employees, afternoon_shift_demand)
                for emp_id in afternoon_assigned_employees:
                    schedule.append({'date': date, 'emp_id': emp_id, 'shift_start': afternoon_shift_start, 'shift_end': afternoon_shift_end, 'role': role})
                    assigned_employees.setdefault(date, []).append(emp_id)

    return schedule

# Output the schedule
def output_schedule(schedule):
    if schedule:
        with open('C:/Users/Carolin/Desktop/schedule.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)  # Default delimiter is comma
            writer.writerow(['Date', 'Employee ID', 'Shift Start Time', 'Shift End Time', 'Role'])
            for entry in schedule:
                # Format datetime objects to display only time
                shift_start_time = entry['shift_start'].strftime('%H:%M')
                shift_end_time = entry['shift_end'].strftime('%H:%M')
                writer.writerow([entry['date'], entry['emp_id'], shift_start_time, shift_end_time, entry['role']])
        print("Schedule generated successfully.")
    else:
        print("No shifts were assigned. Check the availability of employees and other constraints.")

def automated_staff_scheduling_page():
    st.title("Automated Staff Scheduling")

    # Date range selection
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    # Load data from CSV file or your data source
    csv_file_path = "C:/Users/Carolin/Desktop/3yeardata.csv"
    df = pd.read_csv(csv_file_path)

    # Feature engineering: Extract day as a feature
    df['Date'] = pd.to_datetime(df['Date'])  # Convert 'Date' column to datetime
    df['Day'] = df['Date'].dt.day  # Extract day of the month

    # Split the dataset into features (X) and target variable (y)
    X = df[['Day']]
    y = df['Sales']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the XGBoost model
    params = {
        'objective': 'reg:squarederror',  # For regression tasks
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 200
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)

    # Specify start and end date (user input)
    start_date_input = start_date.strftime("%Y-%m-%d")
    end_date_input = end_date.strftime("%Y-%m-%d")

    start_date = pd.to_datetime(start_date_input)
    end_date = pd.to_datetime(end_date_input)

    # Initialize a list to store demand data
    demand_data = []

    # Loop through the range of dates
    current_date = start_date
    while current_date <= end_date:
        new_features = pd.DataFrame({
            'Day': [current_date.day]
        })

        # Make predictions on the new features
        new_features['Predicted_Sales'] = model.predict(new_features[['Day']])

        # Role-Specific Demand Calculation for the new date
        demand_per_role = {'Date': current_date}
        total_demand_hours = 0  # Total demand in hours for the current date

        for role, labor_hours_per_unit in average_labor_hours_per_unit.items():
            # Calculate labor demand in hours for the role
            labor_demand_hours = new_features['Predicted_Sales'] * labor_hours_per_unit

            # Apply role-specific factors to adjust labor demand
            labor_demand_hours *= role_specific_factors[role]

            # Apply day-of-week factors to adjust labor demand
            day_name = current_date.day_name()
            labor_demand_hours *= day_of_week_factors[day_name]

            # Sum up the total labor demand for the current date
            total_demand_hours += labor_demand_hours.sum()

            # Output the labor demand for the role on the current date
            print(f'{role} Labor Demand on {current_date.strftime("%Y-%m-%d")}: {labor_demand_hours.sum()} hours')

            # Store the labor demand for the role for the current date
            demand_per_role[f'{role}_Labor_Demand'] = labor_demand_hours.sum()

        # Append demand data for the current date
        demand_data.append(demand_per_role.copy())

        # Move to the next date
        current_date += pd.Timedelta(days=1)

    # Create a DataFrame from the demand data
    demand_df = pd.DataFrame(demand_data)

    # Rename the columns in the demand DataFrame
    demand_df.rename(columns={
        'Date': 'date',
        'Manager_Labor_Demand': 'manager',
        'Assistant Manager_Labor_Demand': 'assistant_manager',
        'Cook_Labor_Demand': 'cook',
        'Cashier_Labor_Demand': 'cashier',
        'Server_Labor_Demand': 'server'
    }, inplace=True)

    # Save the DataFrame to a CSV file with the updated column names
    demand_df.to_csv("C:/Users/Carolin/predicted_demand.csv", index=False)

    predicted_demand = load_csv('C:/Users/Carolin/predicted_demand.csv')
    role_skills = load_csv('C:/Users/Carolin/Desktop/role_skills.csv')
    employee_details = load_csv('C:/Users/Carolin/Desktop/employee_details_data.csv')

    employees = prepare_employee_data(employee_details)
    roles = prepare_role_skills(role_skills)

    schedule = generate_schedule(predicted_demand, employees, shift_timings)

    # Convert the schedule to DataFrame
    schedule_df = pd.DataFrame(schedule)
    schedule_df['Shift Start Time'] = schedule_df['shift_start'].dt.strftime('%H:%M')
    schedule_df['Shift End Time'] = schedule_df['shift_end'].dt.strftime('%H:%M')
    # Display the schedule
    st.subheader("Generated Schedule:")
    st.dataframe(schedule_df)

if __name__ == "__main__":
    automated_staff_scheduling_page()
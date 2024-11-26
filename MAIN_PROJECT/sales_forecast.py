import streamlit as st
import pandas as pd
import calendar
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Function to predict sales for a given date range
def predict_sales(start_date, end_date, model):
    # Initialize a list to store demand data
    demand_data = []
    # Loop through the range of dates
    current_date = start_date
    while current_date <= end_date:
        # Make predictions on the current date
        predicted_sales = model.predict([[current_date.day]])[0]  # Note the change here
        # Append the predicted sales to the demand_data list
        demand_data.append({
            'Date': current_date,
            'Predicted_Sales': predicted_sales
        })
        # Move to the next day
        current_date += pd.Timedelta(days=1)
    # Convert demand_data to a DataFrame
    predicted_sales_df = pd.DataFrame(demand_data)
    # Convert the 'Date' column to datetime type
    predicted_sales_df['Date'] = pd.to_datetime(predicted_sales_df['Date'])
    return predicted_sales_df

def train_model():
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
    # Define the hyperparameters for the XGBoost model
    params = {
        'objective': 'reg:squarederror',  # For regression tasks
        'max_depth': 5,
        'learning_rate': 0.05,
        'n_estimators': 200
    }
    # Train the XGBoost model
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model




def sales_forecast_page():
    # Load data and train the model
    model = train_model()
    
    st.title('Sales Forecast and Analysis')
    
    # Create sub-pages
    sub_page = st.radio("Select a sub-page", ("Daily Sales Forecast", "Monthly Sales Forecast", "Weekly and Day of Week Sales Forecast", "Analysis Report"))
    
    if sub_page == "Daily Sales Forecast":
        st.subheader("Daily Sales Forecast")
        # User input for start and end dates
        start_date_input = st.date_input("Enter start date:")
        end_date_input = st.date_input("Enter end date:")
        
        if start_date_input < end_date_input:
            # Generate predicted sales data for the user-defined date range
            predicted_sales_df = predict_sales(start_date_input, end_date_input, model)
            
            # Display the predicted sales for each date
            st.subheader("Predicted Sales")
            st.write(predicted_sales_df)
            
            # Plot predicted sales data
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(predicted_sales_df['Date'], predicted_sales_df['Predicted_Sales'])
            ax.set_xlabel('Date')
            ax.set_ylabel('Predicted Sales')
            ax.set_title('Predicted Sales')
            ax.grid(True)
            st.pyplot(fig)
        else:
            st.error('Error: End date must be after start date.')
    
    elif sub_page == "Monthly Sales Forecast":
        st.subheader("Monthly Sales Forecast")
        # User input for year
        year_input = st.number_input("Enter the year:", min_value=2020, max_value=2030, step=1)
        
        # Generate predicted sales data for the entire year
        start_date = pd.to_datetime(f"{year_input}-01-01")
        end_date = pd.to_datetime(f"{year_input}-12-31")
        predicted_sales_df = predict_sales(start_date, end_date, model)
        
        # Display the predicted sales for each month
        st.subheader("Predicted Sales by Month")
        predicted_sales_df['Month'] = predicted_sales_df['Date'].dt.month
        monthly_sales = predicted_sales_df.groupby('Month')['Predicted_Sales'].sum().reset_index()
        st.write(monthly_sales)
        
        # Plot monthly predicted sales
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(monthly_sales['Month'], monthly_sales['Predicted_Sales'])
        ax.set_xlabel('Month')
        ax.set_ylabel('Predicted Sales')
        ax.set_title(f'Monthly Predicted Sales for {year_input}')
        ax.set_xticks(monthly_sales['Month'])
        ax.set_xticklabels([calendar.month_abbr[month] for month in monthly_sales['Month']])
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    elif sub_page == "Weekly and Day of Week Sales Forecast":
        st.subheader("Weekly and Day of Week Sales Forecast")
        # User input for month and year
        month_input = st.number_input("Enter the month:", min_value=1, max_value=12, step=1)
        year_input = st.number_input("Enter the year:", min_value=2020, max_value=2030, step=1)
        
        # Generate predicted sales data for the specified month and year
        start_date = pd.to_datetime(f"{year_input}-{month_input}-01")
        end_date = start_date + pd.offsets.MonthEnd()
        predicted_sales_df = predict_sales(start_date, end_date, model)
        
        # Display the predicted sales for each week
        st.subheader("Predicted Sales by Week")
        predicted_sales_df['Week'] = predicted_sales_df['Date'].dt.isocalendar().week
        weekly_sales = predicted_sales_df.groupby('Week')['Predicted_Sales'].sum().reset_index()
        st.write(weekly_sales)
        
        # Plot weekly predicted sales
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(weekly_sales['Week'], weekly_sales['Predicted_Sales'])
        ax.set_xlabel('Week')
        ax.set_ylabel('Predicted Sales')
        ax.set_title(f'Weekly Predicted Sales for {calendar.month_name[month_input]} {year_input}')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Display the predicted sales for each day of the week
        st.subheader("Predicted Sales by Day of Week")
        predicted_sales_df['Day of Week'] = predicted_sales_df['Date'].dt.day_name()
        sales_by_day = predicted_sales_df.groupby('Day of Week')['Predicted_Sales'].mean().reset_index()
        st.write(sales_by_day)
        
        # Plot predicted sales by day of the week
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(sales_by_day['Day of Week'], sales_by_day['Predicted_Sales'])
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Average Predicted Sales')
        ax.set_title(f'Predicted Sales by Day of Week for {calendar.month_name[month_input]} {year_input}')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    elif sub_page == "Analysis Report":
        st.subheader("Analysis Report")
        # User input for year
        year_input = st.number_input("Enter the year:", min_value=2020, max_value=2030, step=1)
        
        # Generate predicted sales data for the entire year
        start_date = pd.to_datetime(f"{year_input}-01-01")
        end_date = pd.to_datetime(f"{year_input}-12-31")
        predicted_sales_df = predict_sales(start_date, end_date, model)
        
        # Display analysis reports
        total_predicted_sales = predicted_sales_df['Predicted_Sales'].sum()
        st.write(f"Total Predicted Sales for {year_input}: {total_predicted_sales:.2f}")
        
        average_predicted_sales = predicted_sales_df['Predicted_Sales'].mean()
        st.write(f"Average Predicted Sales per Day for {year_input}: {average_predicted_sales:.2f}")
        
       # predicted_sales_growth_rate = ((predicted_sales_df['Predicted_Sales'].iloc[-1] - predicted_sales_df['Predicted_Sales'].iloc[0]) / predicted_sales_df['Predicted_Sales'].iloc[0]) * 100
        #st.write(f"Predicted Sales Growth Rate for {year_input}: {predicted_sales_growth_rate:.2f}%")
        
        max_sales_month = predicted_sales_df.groupby(predicted_sales_df['Date'].dt.month)['Predicted_Sales'].sum().idxmax()
        st.write(f"Month with Maximum Sales in {year_input}: {calendar.month_name[max_sales_month]}")
        
        max_sales_date = predicted_sales_df.loc[predicted_sales_df['Predicted_Sales'].idxmax(), 'Date']
        st.write(f"Date with Maximum Sales in {year_input}: {max_sales_date.strftime('%Y-%m-%d')}")
        
        max_sales_day = predicted_sales_df.groupby(predicted_sales_df['Date'].dt.day_name())['Predicted_Sales'].sum().idxmax()
        st.write(f"Day of Week with Most Predicted Sales in {year_input}: {max_sales_day}")
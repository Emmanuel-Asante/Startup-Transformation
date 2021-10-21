# Import modules
import codecademylib3
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler

# load in financial data
financial_data = pd.read_csv('financial_data.csv')

# code goes here

# Output the first five rows of financial_data
print(financial_data.head())

# Store "Month" column of financial_data in a variable named month
month = financial_data["Month"]

# Store "Revenue" column of financial_data in a variable named revenue
revenue = financial_data["Revenue"]

# Store "Expenses" column of financial_data in a variable named expenses
expenses = financial_data["Expenses"]

# create a plot of revenue over the past six months
plt.plot(month, revenue)

# Add label on x-axis
plt.xlabel('Month')

# Add label on y-axis
plt.ylabel('Amount ($)')

# Add title to chart
plt.title('Revenue')

# Display the chart
plt.show()

# Clear the entire current figure with all its axes
plt.clf()

# Create a plot of monthly expenses over the past 6 months
plt.plot(month, expenses)
plt.xlabel('Month')
plt.ylabel('Amount ($)')
plt.title('Expenses')
plt.show()

# How are monthly expenses changing over time?
print('\nMonthly expenses are increasing exponentially\n')

# Use pandas to read in expenses.csv and store it in a variable called expense_overview
expense_overview = pd.read_csv('expenses.csv')

# Display the first seven rows of expense_overview
print(expense_overview.head(7))

# Store the "Expense" column in expense_overview in a variable called expense_categories
expense_categories = expense_overview['Expense']

# Store the "Proportion" column in expense_overview in a variable called proportions
proportions = expense_overview['Proportion']

# Create a pie chart of expense_overview dataset
expense_categories = ['Salaries', 'Advertising', 'Office Rent', 'Other']
proportions = [0.62, 0.15, 0.15, 0.08]
plt.clf()
plt.pie(proportions, labels = expense_categories)
plt.title('Expense Categories')
plt.axis('Equal')
plt.tight_layout()
plt.show()
plt.clf()

# Which expense categories make up most of the data, and which ones aren’t so significant?
print('Salaries, Advertising and Office Rent  categories make up most of the data\n')

print('Food, Supples, Utilities and Equipment categories aren\'t so significant\n')

# If the company wants to cut costs in a big way, which category do you think they should focus on?
expense_cut = 'Salaries'

# Load employees.csv and store it in a variable called employees
employees = pd.read_csv('employees.csv')

# Print the first few rows of the data
print(employees.head())

# Sort the employees data frame (in ascending order) by the Productivity column and store the result in a variable called sorted_productivity
sorted_productivity = employees.sort_values(by = ['Productivity'])

# Print sorted_productivity
print(sorted_productivity)

# Store the first 100 rows of sorted_productivity in a new variable called employees_cut
employees_cut = sorted_productivity.head(100)

# Print out employees_cut
print(employees_cut)

# The right data transformation method
transformation = 'standardization'

# Create a variable called commute_times that stores the Commute Time column
commute_times = employees['Commute Time']

# Create a variable called commute_times_log that stores a log-transformed version of commute_times
commute_times_log = np.log(commute_times)

# Print out a descriptive statistics for commute_times
print(commute_times.describe())

# What are the average and median commute times?
print('\nThe average and median commute times are 33 and 31 minutes, respectively\n')

# Plot a histogram of commute_times
plt.hist(commute_times_log)
plt.title("Employee Commute Times")
plt.xlabel("Commute Time")
plt.ylabel("Frequency")
plt.show()

# What do you notice about the shape of the data?
print('The commute time data appears to be right-skewed')

# Apply standardization to the employees data using StandardScaler() from sklearn
scaler = StandardScaler()

# Standardized emplyees salary data
standardized_employees_salary = scaler.fit_transform(employees[['Salary']])

# Print out standardized_employees_salary
print(standardized_employees_salary)

# Standardized emplyees productivity data
standardized_employees_productivity = scaler.fit_transform(employees[[ 'Productivity']])

# Display standardized_employees_productivity
print(standardized_employees_productivity)

# Standardized emplyees commute time data
standardized_employees_commute_time = scaler.fit_transform(employees[['Commute Time']])

# Output standardized_employees_commute_time
print(standardized_employees_commute_time)

# Return a NumPy ndarray representing the columns "Salary" and "Productivity" into a variable called data
data = employees[['Salary', 'Productivity']].to_numpy()

# Transfrom data and store it in a variable called standardize_data
standardize_data = scaler.fit_transform(data)

# Plot a scatter plot
plt.clf()
x = standardize_data[:,0]
y = standardize_data[:,1]
plt.scatter(x,y)
plt.show()
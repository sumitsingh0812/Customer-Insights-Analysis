
# 1. Importing necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# 2. Define file paths
filepath = r'C:\Users\Admin\Desktop\Retail_Data_Transactions.csv'
file2path = r"C:\Users\Admin\Desktop\Retail_Data_Response.csv"

# 3. Read the CSV files into DataFrames
df1 = pd.read_csv(filepath)
df2 = pd.read_csv(file2path)
print(df1)
print(df2)

# 4. Merge the DataFrames
merged_df = df1.merge(df2, on='customer_id', how='left')

# 5. Print the merged DataFrame
print("Merged DataFrame:")
print(merged_df)

# 6. Display features and their data types
df3 = merged_df.dtypes
print("\nData types of features:")
print(df3)

# 7. Display shape of the DataFrame
df4 = merged_df.shape
print("\nShape of DataFrame:")
print(df4)

# 8. Display the first 5 rows
df5 = merged_df.head()
print("\nFirst 5 rows of the DataFrame:")
print(df5)

# 9. Display the last 5 rows
df6 = merged_df.tail()
print("\nLast 5 rows of the DataFrame:")
print(df6)

# 10. Check for missing values
df_missingvalues = merged_df.isnull()
print("\nMissing values in DataFrame:")
print(df_missingvalues)

# 11. Count the missing values
df_count = merged_df.isnull().sum()
print("\nCount of missing values:")
print(df_count)

# 12. Check if any missing value exists
missing_value_exist = merged_df.isnull().values.any()
print("\nDo missing values exist?:")
print(missing_value_exist)

# 13. Drop rows with missing values and display the result
drop = merged_df.dropna()
print("\nDataFrame after dropping rows with missing values:")
print(drop)

# 14. Calculate and display the mean of 'tran_amount'
mean = merged_df['tran_amount'].mean()
print("\nMean of transaction amounts:")
print(mean)

# 15. Convert 'trans_date' column to datetime
merged_df['trans_date'] = pd.to_datetime(merged_df['trans_date'], errors='coerce')
print("\nDataFrame with 'trans_date' converted to datetime:")
print(merged_df['trans_date'])

# 16. Extract month from 'trans_date' and create a new column 'month'
# Ensure no null values in 'trans_date' after conversion
merged_df = merged_df.dropna(subset=['trans_date'])
merged_df['month'] = merged_df['trans_date'].dt.month
print("\nDataFrame with 'month' extracted from 'trans_date':")
print(merged_df)

# Verify the presence of the 'month' column
if 'month' in merged_df.columns:
    print("\nThe 'month' column has been created successfully.")
else:
    print("\nThe 'month' column was not created successfully.")

# 17. Identify outliers using Z-scores
zscores = np.abs(stats.zscore(merged_df['tran_amount'].dropna()))  # Drop NaN values for z-score calculation
threshold = 3
outlier = zscores > threshold

print("\nZ-scores of outliers:")
print(zscores[outlier])
print("\nOutlier values:")
print(merged_df.loc[outlier, 'tran_amount'])

# 18. Find the top 3 months with the highest transaction amounts
msales = merged_df.groupby('month')['tran_amount'].sum()
msales = msales.sort_values(ascending=False).reset_index().head(3)
print("\nTop 3 months with the highest transaction amounts:")
print(msales)

#19. Find the top 3 months with the lowest transaction amounts
msales= merged_df.groupby('month')['tran_amount'].sum()
msales1 = msales.sort_values(ascending=False).reset_index().tail(3)
print("\nTop 3 months with the lowest transaction amounts:")
print(msales1)

#20 which customerid has the maximum orders
#first creating a new data frame of count

count = merged_df['customer_id'].value_counts().reset_index()
count.columns = ['customer_id', 'count']
print("\nCustomer ID having the maximum orders counts:")
print(count)
#

maxsale = count.sort_values('count', ascending=False).reset_index(drop=True).head(3)

print(maxsale)



#21 which customerid has the min orders

minsale = count.sort_values('count', ascending=True).reset_index(drop=True).tail(3)

print(minsale)
#plotting the graphs for following
sns.set(style='darkgrid')
sns.barplot(x='customer_id',y='count',data=maxsale)
plt.title('top 3 orders by customer_id')
plt.show()


horder = merged_df.groupby('customer_id')['tran_amount'].sum().reset_index()
print("\nTotal transaction amount per customer:")
print(horder)

# Find the top 10 customers with the highest transaction amounts
top10 = horder.sort_values(by='tran_amount', ascending=False).head(10)
print("\nTop 10 customers with the highest transaction amounts:")
print(top10)                                         
# graph

sns.set(style='darkgrid')
sns.barplot(x='customer_id',y='tran_amount',data=top10)
plt.title('top 10 customers with the highest transaction amounts ')
plt.show()


#TIME SERIES ANALYSIS

merged_df['month_year'] = merged_df['trans_date'].dt.to_period('M')
print("\nDataFrame with 'month_year' column:")
print(merged_df)

# now we are seeing the graph of month year and  trans amount
#first we are doing groupby


# Group by 'month_year' and sum 'tran_amount'
myt = merged_df.groupby('month_year')['tran_amount'].sum()

# Convert PeriodIndex to Timestamp
myt.index = myt.index.to_timestamp()

# Reset index to convert Series to DataFrame
myt = myt.reset_index()

# Print the DataFrame
print(myt)

# Plotting
sns.set(style='darkgrid')
plt.figure(figsize=(12, 6))
sns.lineplot(x='month_year', y='tran_amount', data=myt)
plt.title('Month-Year vs Transaction Amount')
plt.xlabel('Month-Year')
plt.ylabel('Transaction Amount')
plt.xticks(rotation=45)
plt.show()

###################################

#track most recent order
recency=merged_df.groupby('customer_id')['trans_date'].max()# max means the maximum value
#customer who buy frequently
frequency=merged_df.groupby('customer_id')['trans_date'].count()
#monetary
monatary=merged_df.groupby('customer_id')['tran_amount'].sum()

#making a new dataframe
newdf={'recency':recency,'frequency':frequency,'monatary':monatary}
dframe=pd.DataFrame(newdf)
print(dframe)


dframe['recency'] = pd.to_datetime(dframe['recency'])

def sc(row):
    # Calculate the number of days from a reference date
    reference_date = pd.Timestamp('1970-01-01')
    days_since_reference = (row['recency'] - reference_date).days


    # Use the days_since_reference for comparison
    if row['recency'].year > 2012 and row['frequency'] > 15 and days_since_reference > 1000:
        return 'P0'
    elif 2011 < row['recency'].year < 2012 and 10 < row['frequency'] < 15 and 500 < row['monetary'] < 1000:
        return 'P1'
    else:
        return 'P2'

dframe['segment'] = dframe.apply(sc, axis=1)


print(dframe)

################################

#count the number of churned customer
ccustomer=merged_df['response'].value_counts()

ccustomer.plot(kind="bar")
plt.show()

###################

top15=monatary.sort_values(ascending=False).head(15).index
print(top15)



# Save merged_df to CSV file
merged_df.to_csv(r'C:\Users\Admin\Desktop\Data1.csv', index=False)
print("Merged DataFrame saved to 'Data1.csv'.")

# Save rfm_df to CSV file
dframe.to_csv(r'C:\Users\Admin\Desktop\Data2.csv', index=False)
print("RFM DataFrame saved to 'Data2.csv'.")


# 1. Save the cleaned dataset as CSV
cleaned_filepath_csv = r'C:\Users\Admin\Documents\Cleaned_Online_Retail_Data.csv'
df1.to_csv(cleaned_filepath_csv, index=False)

print(f"Cleaned data saved as CSV at: {cleaned_filepath_csv}")


    





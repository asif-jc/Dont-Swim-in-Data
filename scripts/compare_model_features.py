import pandas as pd

feature_set_one = pd.read_csv("/Users/asif/Documents/Forecasting Microbial Contamination (Master's Project)/Code/Don't Swim in Data/data/processed/temporal_features.csv")
feature_set_two = pd.read_csv("/Users/asif/Documents/Forecasting Microbial Contamination (Master's Project)/Code/Cleaned-Machine-Learning-Pipeline-Safeswim/ml_pipeline/Experiment Results/Matrix Framework/temporal_data_old.csv")

# Set 'DateTime' as the index and sort the rows by DateTime
df1 = feature_set_one.set_index('DateTime').sort_index()
df2 = feature_set_two.set_index('DateTime').sort_index()

# Sort the columns alphabetically to avoid issues due to ordering differences
df1 = df1.sort_index(axis=1)
df2 = df2.sort_index(axis=1)

# (Optional) Cast specific columns to the same type if needed to avoid false differences
cols_to_cast = ['HOLIDAY_FLAG', 'MONTH', 'WEEK', 'DAY_OF_WEEK', 'WEEKEND', 'TIME_OF_DAY']
for col in cols_to_cast:
    df1[col] = df1[col].astype(float)
    df2[col] = df2[col].astype(float)

# Use the DataFrame.compare() method to find differences
differences = df1.compare(df2)

print("Differences between the two DataFrames:")
print(differences)

# To see which rows (based on DateTime) have differences:
if not differences.empty:
    diff_rows = differences.index.unique()
    print("\nRows with differences (DateTime values):")
    print(diff_rows)
else:
    print("No differences found between the two DataFrames.")
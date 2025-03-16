import pandas as pd

feature_set_one = pd.read_csv("/Users/asif/Documents/Forecasting Microbial Contamination (Master's Project)/Code/Don't Swim in Data/data/processed/temporal_features.csv")
feature_set_two = pd.read_csv("/Users/asif/Documents/Forecasting Microbial Contamination (Master's Project)/Code/Cleaned-Machine-Learning-Pipeline-Safeswim/ml_pipeline/Experiment Results/Matrix Framework/temporal_data_old.csv")

print(feature_set_one)
print(feature_set_two)

feature_set_one.info()
feature_set_two.info()
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, LabelEncoder

df = pd.read_csv("diabetes_data.csv", sep=";")
normalized_df = df.copy(deep=True)

# df["class"].value_counts()
#
# print("No fraud samples make up", round(df["class"].value_counts()[0]/len(df) * 100,2), "% of the dataset.")
# print("Fraud samples make up", round(df["class"].value_counts()[1]/len(df) * 100,2), "% of the dataset.")
#
# sns.countplot(x="class", data=df)

# scaler = MinMaxScaler()
# normalized_data = scaler.fit_transform(df)

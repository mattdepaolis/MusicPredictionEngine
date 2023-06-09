import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv("train.csv")

profile = ProfileReport(df, title="Exploratory Data Analysis for Deezer Music Data")
profile.to_file("eda.html")

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools

data = pd.read_csv("Iris.csv")

x1 = data.iloc[:,1].values
y1 = data.iloc[:,2].values

df = pd.DataFrame({'x': x1,'y': y1})
print(df)

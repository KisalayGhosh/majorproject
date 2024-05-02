import pandas as pd
import xarray as xr
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = xr.open_dataset('myfile.nc')


df = data.to_dataframe().reset_index()

max_value = df['APCP_sfc'].max()

min_value = df['APCP_sfc'].min()

print("Maximum value of APCP_sfc:", max_value)
print("Minimum value of APCP_sfc:", min_value)

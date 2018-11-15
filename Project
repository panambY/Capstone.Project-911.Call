# This project is in the Kaggle Kernel: 
https://www.kaggle.com/panamby/01-capstone-projet-911-calls

# For this Capstone Project we will be analyzing some 911 call data from Kaggle. 

# Just go along with this notebook and try to complete some tasks using my Python and Data Science skills!
# First of all, I will import the two libraries that will help me analyze the 911 data. 
# One of them is the poweful Pandas.🐼

import numpy as np
import pandas as pd

# Secondly, I will import the libraries for data visualization.
# For a greater practicality, I will tap a code which will enable to watch the graphs after coding them.

import matplotlib.pyplot as plt
import seaborn as sns
% matplotlib inline

# Now I will read the file that I will analyze and visualize.

df = pd.read_csv('../input/911.csv')

# Let's extract some informations from this file.

df.info()   # We can see that there are more than 300,000 entries and 9 columns. 
            # There are numbers (floats and integers) and strings.
            
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 326425 entries, 0 to 326424
Data columns (total 9 columns):
lat          326425 non-null float64
lng          326425 non-null float64
desc         326425 non-null object
zip          286835 non-null float64
title        326425 non-null object
timeStamp    326425 non-null object
twp          326310 non-null object
addr         326425 non-null object
e            326425 non-null int64
dtypes: float64(3), int64(1), object(5)
memory usage: 22.4+ MB

# Let's check the head!

df.head()   # The data contains the following fields:

            # lat : String variable, Latitude
            # lng: String variable, Longitude
            # desc: String variable, Description of the Emergency Call
            # zip: String variable, Zipcode
            # title: String variable, Title
            # timeStamp: String variable, YYYY-MM-DD HH:MM:SS
            # twp: String variable, Township
            # addr: String variable, Address
            # e: String variable, Dummy variable (always 1)
            

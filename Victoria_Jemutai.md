#  HIV DATA ANALYSIS

### PART A 

#### OVERVIEW

In the first part of the analysis we're provided with a dataset from the World Health Organization (WHO) Global Observatory, containing data on people living with HIV at the country level from 2000 to 2023.

From the data, we can analyze infection trends, inform healthcare service planning, and addressing persistent issues like stigma and inequity. 
Particularly for this task, I will use the data to track HIV infections in countries globally and WHO regions. 

I will then conclude with findings based on the visualizations plotted.

### Data Analysis 

The analysis will have two parts:
1. Visualization of HIV cases trends in countries that contribute to 75% of the global burden.

2. Visualization of HIV cases trends in countries that contribute to 75% of the global burden in each WHO region.

The analysis will begin with cleaning data, using pandas library to perform data imputation, matplotlib and seaborn for visualizations.

#### Data Cleaning 

```{python}
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm



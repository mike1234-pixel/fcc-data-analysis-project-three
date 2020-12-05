import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')
# global-setting: show all columns in df.head()
# ------------------------------------------
# pd.set_option('display.max_columns', None) 
# pd.set_option('display.max_rows', None) 

# Add 'overweight' column
# BMI = weight in kg / square of height in m
# > .25 is overweight 
df['overweight'] = (df['weight']) /((df['height'] / 10) * (df['height'] / 10)) 
df.loc[df['overweight'] > 0.25, 'overweight'] = 1
df.loc[df['overweight'] <= 0.25, 'overweight'] = 0

df.overweight = df.overweight.apply(int) # change overweight values from 1.0/0.0 to single ints 1 or 0 

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholestorol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1. 
df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['gluc'] == 1, 'gluc'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1
df.loc[df['gluc'] > 1, 'gluc'] = 1 

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    # convert df to long format
    # group your data in a way that you can later pull a count of value variables corresponding to cardio in the next step
    long_form_df = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    # split the data according to whether cardio is 1 or 0, then count every value under the the long form variable column to create the data set for each plot, for each variable  ('cholesterol', 'gluc' etc)there will be a 0s and 1s bucket
    df_cat = pd.DataFrame(
    long_form_df.groupby(['cardio', 'variable',
                    'value'])['value'].count()).rename(columns={
                        'value': 'total'
                    }).reset_index()

    # Draw the catplot with 'sns.catplot()'
    cat_plot = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar')

    fig = cat_plot.fig

    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
  #Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi'])
              & (df['height'] >= df['height'].quantile(0.025))
              & (df['height'] <= df['height'].quantile(0.975))
              & (df['weight'] >= df['weight'].quantile(0.025))
              & (df['weight'] <= df['weight'].quantile(0.975))]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap with 'sns.heatmap()'
    heat_map = sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, cmap='coolwarm', vmin=-0.1, vmax=0.25, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
    # 1 is perfect positive corr, -1 is perfect negative corr.
    # In this context dark red indicates fairly strong positive corr, e.g. between 'alco' and 'smoke', or 'height' and 'gender'
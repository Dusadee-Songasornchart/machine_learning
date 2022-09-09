import pandas as pd

df = pd.DataFrame({
    'Name': ['Joan', 'Matt', 'Jeff', 'Melissa', 'Devi'],
    'Gender': ['Female', 'Male', 'Male', 'Female', 'Female'],
    'House Type': ['Apartment', 'Detached', 'Apartment', None, 'Semi-Detached']
    })

print(pd.get_dummies(df['Gender']))
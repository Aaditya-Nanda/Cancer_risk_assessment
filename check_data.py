import pandas as pd

df = pd.read_csv('data/raw/cancer_patient_data.csv')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print()
print('Target distribution:')
print(df['Level'].value_counts())
print()
print('First 3 rows:')
print(df.head(3))

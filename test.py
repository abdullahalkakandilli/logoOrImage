
import pandas as pd


df = pd.read_csv(r'C:\Users\alka\Masaüstü\imagesT.csv')

for index, row in df.iterrows():
    print(row['Logo'])
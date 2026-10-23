import pandas as pd
with open('../data/moleculeIndex.json', 'r') as f:
  df = pd.read_json(f)
  
df.to_csv('../data/toExcel.csv', encoding='utf-8', index=False)
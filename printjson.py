import json
import pandas as pd
with open('moleculeIndex.json', 'r') as f:
  df = pd.read_json(f)
  
df.to_csv('toExcel.csv', encoding='utf-8', index=False)
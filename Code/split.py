import pandas as pd
import os

# 加载原始数据
file_path = "Metrics.csv"
output_dir = "split_results"


os.makedirs(output_dir, exist_ok=True)


df = pd.read_csv(file_path)


chunk_size = 2500
for i, chunk in enumerate(range(0, df.shape[0], chunk_size)):
    output_file = os.path.join(output_dir, f"Results_part_{i}.csv")
    df.iloc[chunk:chunk + chunk_size].to_csv(output_file, index=False)
    print(f"Saved {output_file}")

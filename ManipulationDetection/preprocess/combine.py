import pandas as pd
dir = "./output/"
filename = []
for i in range(1,17):
    filename.append(str(i)+".csv")
combined_csv = pd.concat([pd.read_csv(dir+f) for f in filename],sort=False)
combined_csv.to_csv( dir+"combined.csv", index=False, encoding='utf-8-sig')
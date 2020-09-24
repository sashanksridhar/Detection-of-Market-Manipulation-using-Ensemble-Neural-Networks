from sklearn import preprocessing
import pandas as pd
import csv

with open("./output/combined.csv",'r') as r:
    X = []
    Y = []
    header = []
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c==0:
            header.extend(row)
            c+=1
            continue
        x = []
        for i in range(0,len(row)-1):
            x.append(float(row[i]))
        X.append(x)
        Y.append(row[len(row)-1])
    print(len(X))
    print(len(Y))
    print(X)
    header.remove('Label')
    df = pd.DataFrame(X,columns =header )

    # normal = preprocessing.normalize(df,axis=0)
    maxObj = df.max()
    for label,content in df.items():
        if float(maxObj[label])!=float(0):
            df[label] = df[label].div(float(maxObj[label]))
    l = df.values.tolist()
    for i in range(0,len(Y)):
        for j in range(0,len(Y[i])):
            l[i].append(Y[i][j])

    with open("./preprocess/normalized.csv",'w') as w:
        writer = csv.writer(w,lineterminator='\n')
        header.append('Label')
        writer.writerow(header)
        for i in range(0,len(l)):
            writer.writerow(l[i])
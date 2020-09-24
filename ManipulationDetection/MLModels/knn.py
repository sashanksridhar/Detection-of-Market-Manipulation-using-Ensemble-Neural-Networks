from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import csv
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

Xi = []
Y = []

with open("./preprocess/normalized.csv", 'r') as r:
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c==0:
            c+=1
            continue

        for j in range(0,len(row)):
            row[j] = float(row[j])
        Xi.append(row)

train ,test = train_test_split(Xi,test_size=0.25)
X = []
Xt = []
Yt = []
for i in train:
    X.append(i[:len(i)-1])
    Y.append(i[len(i)-1:][0])
for i in test:
    Xt.append(i[:len(i)-1])
    Yt.append(i[len(i) - 1:][0])
print(len(Y))
print(len(Yt))


classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X, Y)
y_pred = classifier.predict(Xt)
count = 0
j = 0
for i in y_pred:
    if int(i) == int(Yt[j]):
        count = count+1
    j+=1
print(count)
print((count/len(Yt))*100)

fig, ax1 = plt.subplots()
pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
ax1.scatter(data2D[:,0],data2D[:,1])
ax1.legend()
plt.show()

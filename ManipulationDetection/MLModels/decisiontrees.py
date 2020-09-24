from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import csv
from sklearn.tree import export_graphviz
import pydotplus
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
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X,Y)

#Predict the response for test dataset
y_pred = clf.predict(Xt)
count = 0
j = 0
for i in y_pred:
    if int(i) == int(Yt[j]):
        count = count+1
    j+=1
print(count)
print((count/len(Yt))*100)

dot_data = export_graphviz(clf)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png("tree.png")
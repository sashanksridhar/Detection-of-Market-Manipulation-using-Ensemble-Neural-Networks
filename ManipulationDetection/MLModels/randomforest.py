import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix
Xi = []
Y = []

with open("./preprocess/normalized.csv", 'r') as r:
    c = 0
    reader = csv.reader(r)
    for row in reader:
        if c==0:
            c+=1
            continue

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

n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
acc = []
for estimator in n_estimators:
    model = RandomForestClassifier(n_estimators=estimator,
                                   bootstrap = True,
                                   max_features = 'sqrt')
    # Fit on training data
    model.fit(X, Y)
    # Prediction
    predicted = model.predict(Xt)
    count = 0
    j = 0
    for i in predicted:
        if int(i) == int(Yt[j]):
            count = count+1
        j+=1
    print(count)
    print((count/len(Yt))*100)
    acc.append((count/len(Yt))*100)
    print("=== Confusion Matrix ===")
    print(confusion_matrix(Yt, predicted))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(Yt, predicted))
    print('\n')

tree_list = np.array(n_estimators)
accuracy_percent = np.array(acc)
plt.plot(tree_list,accuracy_percent)
plt.xlabel('Number of trees')
plt.ylabel('Percent of accuracy')
plt.title('Varation of accuracy with trees')
plt.grid(True)
plt.savefig("rf1.png")
plt.show()

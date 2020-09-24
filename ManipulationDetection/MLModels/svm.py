from sklearn import svm
from sklearn.model_selection import train_test_split
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
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
#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X, Y)

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

pca = PCA(n_components=2).fit(X)
data2D = pca.transform(X)
clf = svm.SVC(decision_function_shape='ovo')
clf.fit(data2D,Y)
import numpy
plot_decision_regions(X=data2D,
                      y=numpy.array(Y).astype(numpy.integer),
                      clf=clf,
                      legend=2)


plt.title('SVM Decision Region Boundary', size=16)
plt.show()
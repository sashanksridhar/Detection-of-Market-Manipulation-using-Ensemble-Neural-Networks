import csv
from sklearn.model_selection import train_test_split
import numpy
from keras.utils.vis_utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
# load the dataset
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
print(X)
print(len(X))
print(len(Y))
# split into input (X) and output (y) variables

# Build the neural network
model = Sequential()
model.add(Dense(20, input_dim=20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(20, activation='relu'))

model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))

model.add(Dense(1, activation='sigmoid'))
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset

Xout = numpy.array(X)
model.fit(Xout, Y, epochs=500)
# evaluate the keras model
_, accuracy = model.evaluate(Xout, Y)
print('Accuracy: %.2f' % (accuracy*100))
Xtest = numpy.array(Xt)
predicted = model.predict_classes(Xtest)
count = 0
j = 0
for i in predicted:
    if int(i) == int(Yt[j]):
        count = count+1
    j+=1
print(count)
print((count/len(Yt))*100)
model_json = model.to_json()
with open("model_x.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_x.h5")
print("Saved model to disk")


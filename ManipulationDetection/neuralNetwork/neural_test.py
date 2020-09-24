from keras.models import model_from_json
import csv
from sklearn.model_selection import train_test_split
import numpy
json_file = open('./neuralNetwork/model_x.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights('./neuralNetwork/model_x.h5')
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

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
print(len(X))
print(len(Y))

Xtest = numpy.array(Xt)
predicted = loaded_model.predict_classes(Xtest)
count = 0
j = 0
for i in predicted:
    if int(i) == int(Yt[j]):
        count = count+1
    j+=1
print(count)
print((count/len(Yt))*100)
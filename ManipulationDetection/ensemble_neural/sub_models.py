# example of saving sub-models for later use in a stacking ensemble
import csv
from sklearn.model_selection import train_test_split
import numpy
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot



# fit model on dataset
def fit_model(trainX, trainy):
    # define model
    model = Sequential()
    model.add(Dense(20, input_dim=20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))
    model.add(Dense(20, activation='relu'))

    model.add(Dense(12, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit model
    model.fit(trainX, trainy, epochs=500)
    return model

Xi = []
# generate 2d classification dataset
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
Y =[]
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
trainX = numpy.array(X)
testX = numpy.array(Xt)
print(trainX.shape, testX.shape)
# create directory for models

# fit and save models
n_members = 6
for i in range(n_members):
    # fit model
    model = fit_model(trainX, Y)
    # save model
    filename = './ensemble_neural/models_6/' + str(i + 1) + '.h5'
    model.save(filename)
    print('>Saved %s' % filename)

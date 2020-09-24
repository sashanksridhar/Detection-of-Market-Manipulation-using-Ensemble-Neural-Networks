import csv
from sklearn.model_selection import train_test_split
import numpy
import matplotlib.pyplot as plt
from keras.models import load_model

from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import plot_model
from keras.models import Model

from keras.layers import Dense
from keras.layers.merge import concatenate

def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        filename = './ensemble_neural/models_5/' + str(i + 1) + '.h5'
        model = load_model(filename)
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


def define_stacked_model(members):

    for i in range(len(members)):

        model = members[i]
        for layer in model.layers:

            # layer.trainable = False

            layer.name = 'ensemble_' + str(i+1) + '_' + layer.name


    # define multi-headed input
    ensemble_visible = [model.input for model in members]

    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]

    merge = concatenate(ensemble_outputs)
    hidden = Dense(10, activation='relu')(merge)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file='model_graph.png')
    # compile
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fit a stacked model
def fit_stacked_model(model, inputX, inputy):
    # prepare input data
    X = [inputX for _ in range(len(model.input))]

    # model.fit(X, inputy, epochs=300)
    history = model.fit(X, inputy, validation_split=0.33, epochs=500, batch_size=10)
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

def predict_stacked_model(model, inputX):
    X = [inputX for _ in range(len(model.input))]
    return model.predict(X)


Xi = []
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

n_members = 5
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

stacked_model = define_stacked_model(members)

fit_stacked_model(stacked_model, trainX,Y)

yhat = predict_stacked_model(stacked_model, testX)
p = []

for x in numpy.nditer(yhat):
    xi = []


    if x < float(0.5):
        p.append(0)
    else:
        p.append(1)



count = 0
for i in range(0,len(Yt)):

    if Yt[i]==p[i]:
        count +=1

print(count)
print((count/len(Yt))*100)
print("=== Confusion Matrix ===")
print(confusion_matrix(Yt, p))
print('\n')
print("=== Classification Report ===")
print(classification_report(Yt, p))
print('\n')
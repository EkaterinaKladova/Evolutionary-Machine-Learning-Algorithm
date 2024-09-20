import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import activations
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("NY-House-Dataset.csv")
data = data[["BROKERTITLE", "TYPE", "BEDS", "BATH", "PROPERTYSQFT", "STATE", "ADMINISTRATIVE_AREA_LEVEL_2", "LOCALITY", "SUBLOCALITY", "STREET_NAME", "LATITUDE", "LONGITUDE", "PRICE"]]
data["BROKERTITLE"] = data["BROKERTITLE"].astype('category').cat.codes
data["TYPE"] = data["TYPE"].astype('category').cat.codes
data["STATE"] = data["STATE"].astype('category').cat.codes
data["ADMINISTRATIVE_AREA_LEVEL_2"] = data["ADMINISTRATIVE_AREA_LEVEL_2"].astype('category').cat.codes
data["LOCALITY"] = data["LOCALITY"].astype('category').cat.codes
data["SUBLOCALITY"] = data["SUBLOCALITY"].astype('category').cat.codes
data["STREET_NAME"] = data["STREET_NAME"].astype('category').cat.codes

# making classes
# by taking the 75th percentile as our dividing point, we can guarantee there is a 3:1 split in classes
# print(data["PRICE"].quantile(.75))
data["PRICE"] = data["PRICE"].apply(lambda x : 1 if x>1495000 else 0)

train, test = train_test_split(data, test_size = 100)

# -Randomly relabel 5% of the training data from each class, e.g., labeling 5% of the dogs as cats, and vice versa.  This ensures that very high accuracy is not achievable.

trainX = train[["BROKERTITLE", "TYPE", "BEDS", "BATH", "PROPERTYSQFT", "STATE", "ADMINISTRATIVE_AREA_LEVEL_2", "LOCALITY", "SUBLOCALITY", "STREET_NAME", "LATITUDE", "LONGITUDE"]]
trainY = train[["PRICE"]]

testX = test[["BROKERTITLE", "TYPE", "BEDS", "BATH", "PROPERTYSQFT", "STATE", "ADMINISTRATIVE_AREA_LEVEL_2", "LOCALITY", "SUBLOCALITY", "STREET_NAME", "LATITUDE", "LONGITUDE"]]
testY = test[["PRICE"]]

'''END DATA PROCESSING'''

trials = 1
gens = 32 # should be 256

k = 5 # how often to update omega
omega = 1 # scale of mutation (updates automatically)
D = 100 # max for omega
lr = 1.5 # rate of change for omega

trialqs = []
testTrialqs = []
parentTopqs = []
cmTest = None
cmTrain = None
for trial in range(trials):
    # initialize 4 parents
    parents = []
    for i in range(4):
        p = Sequential()
        p.add(Dense(12, input_dim = 12, activation=activations.sigmoid))
        p.add(Dropout(0.1))
        p.add(Dense(24))
        p.add(Dropout(0.1))
        p.add(Dense(1))
        p.compile(loss="mse", metrics=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives'])
        matrix = p.evaluate(trainX, trainY, verbose=0)
        q = (matrix[3] + matrix[4])/(sum(matrix[1:5])) + 10 * (max(0, matrix[3]/(matrix[2]+ matrix[3]) + max(0, matrix[4]/(matrix[1] + matrix[4]))))
        parents.append((p, q))
    parents.sort(key = lambda x: x[1])
    parentTopqs.append(parents[0][1])

    topqs = []
    testqs = []
    
    success = 0 # how many children were better than parents
    gen = 0
    while gen < gens:

        # rounds of omega updates
        counter = 0

        # generate children
        children = []
        for c in range(28): # should be 144
            p1, p1q = random.choice(parents)
            p2, p2q = random.choice(parents)
            # average parents per layer
            lay0w = [(x+y)/2 for (x, y) in zip(p1.layers[0].get_weights()[0], p2.layers[0].get_weights()[0])]
            lay0b = [(x+y)/2 for (x, y) in zip(p1.layers[0].get_weights()[1], p2.layers[0].get_weights()[1])]
            lay2w = [(x+y)/2 for (x, y) in zip(p1.layers[2].get_weights()[0], p2.layers[2].get_weights()[0])]
            lay2b = [(x+y)/2 for (x, y) in zip(p1.layers[2].get_weights()[1], p2.layers[2].get_weights()[1])]
            lay4w = [(x+y)/2 for (x, y) in zip(p1.layers[4].get_weights()[0], p2.layers[4].get_weights()[0])]
            lay4b = [(x+y)/2 for (x, y) in zip(p1.layers[4].get_weights()[1], p2.layers[4].get_weights()[1])]

            # mutate
            # creating mutation methods
            rng = np.random.default_rng()
            mut0w = rng.normal(0, omega, 12)
            mut0b = rng.normal(0, omega, 12)
            mut2w = rng.normal(0, omega, 12)
            mut2b = rng.normal(0, omega, 24)
            mut4w = rng.normal(0, omega, 24)
            mut4b = rng.normal(0, omega, 1)
            # mutate child
            lay0 = [np.array([lay0w[i] + mut0w[i] for i in range(12)]), np.array([lay0b[i] + mut0b[i] for i in range(12)])]
            lay2 = [np.array([lay2w[i] + mut2w[i] for i in range(12)]), np.array([lay2b[i] + mut2b[i] for i in range(24)])]
            lay4 = [np.array([lay4w[i] + mut4w[i] for i in range(24)]), np.array([lay4b[i] + mut4b[i] for i in range(1)])]

            # creating child model
            child =  Sequential()
            child.add(Dense(12, input_dim = 12, activation=activations.sigmoid))
            child.layers[0].set_weights(lay0)
            child.add(Dropout(0.1))
            child.add(Dense(24))
            child.layers[2].set_weights(lay2)
            child.add(Dropout(0.1))
            child.add(Dense(1))
            child.layers[4].set_weights(lay4)
            child.compile(loss="mse", metrics=['TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives'])

            # compute q value
            matrix = child.evaluate(trainX, trainY, verbose=0)
            q = (matrix[3] + matrix[4])/(sum(matrix[1:5])) + 10 * (max(0, matrix[3]/(matrix[2]+ matrix[3]) + max(0, matrix[4]/(matrix[1] + matrix[4]))))
            children.append((child, q))
            if (q < p1q) or (q < p2q):
                success += 1

        # combine children and parents
        children.extend(parents)
        # take top 4
        children.sort(key = lambda x: x[1])
        newGen = children[0:4]

        counter += 1
        gen += 1
        
        if counter % k == 0:
            if 20 * success < k:
                omega = min(2 * omega, D)
            elif 5 * success < k:
                omega = omega * lr
            elif 5 * success > k:
                omega = min(omega/lr, D)

            counter = 0
            success = 0
        
        if gen in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            # recording highest q, train data
            matrix = newGen[0][0].evaluate(trainX, trainY, verbose=0)
            q = (matrix[3] + matrix[4])/(sum(matrix[1:5])) + 10 * (max(0, matrix[3]/(matrix[2]+ matrix[3]) + max(0, matrix[4]/(matrix[1] + matrix[4]))))
            topqs.append(q)
            cmTrain = pd.DataFrame({'Real Cheap': [matrix[2], matrix[3]], 'Real Expensive': [matrix[4], matrix[1]]})
            
            # recording highest q, test data
            matrix = newGen[0][0].evaluate(testX, testY, verbose=0)
            q = (matrix[3] + matrix[4])/(sum(matrix[1:5])) + 10 * (max(0, matrix[3]/(matrix[2]+ matrix[3]) + max(0, matrix[4]/(matrix[1] + matrix[4]))))
            testqs.append(q)
            cmTest = pd.DataFrame({'Real Cheap': [matrix[2], matrix[3]], 'Real Expensive': [matrix[4], matrix[1]]})
        print("GEN", gen, "COMPLETE")
    trialqs.append(topqs)
    testTrialqs.append(testqs)


print("Parent Averages:", np.average(parentTopqs))
print("Parent Standard Deviation:", np.std(parentTopqs))

print("Children Averages:", np.average(trialqs, axis = 0))
print("Children Standard Deviations:", np.std(trialqs, axis = 0))

# plotting train/test qs over trials
childDF = pd.DataFrame({"qs": np.average(trialqs, axis = 0), "testqs": np.average(testTrialqs, axis = 0), "gens": np.log([1, 2, 4, 8, 16, 32])})

plt.plot("gens", "qs", data=childDF)
plt.plot("gens", "testqs", data=childDF)
plt.xlabel("log(number of generations)")
plt.ylabel("q average")
plt.show()

print("Training Data Confusion Matrix")
print(cmTrain)
print("Testing Data Confusion Matrix")
print(cmTest)

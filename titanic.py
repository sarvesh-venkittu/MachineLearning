#basic imports
import math
import pandas as pd
import csv
import statistics

#sci-kit learn imports
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

#initialization of models to be used for analysis
model1 = Perceptron()
model2 = svm.SVC()
model3 = KNeighborsClassifier(n_neighbors=1)
model4 = GaussianNB()

#reads data file, ommitting columns that are probably irrelevant
columns_to_be_removed = ['Name', 'Ticket', 'Cabin']
data = pd.read_csv('train.csv').drop(columns_to_be_removed, axis = 'columns')

#separates data into evidence (list of lists) and labels (list)
labels = data.iloc[:, 1].values.tolist()
evidence = data.iloc[:, 2:].values.tolist()

#converts evidence to floating point numbers and integers
def standardize(evidence):
    for row in evidence:
        for i in range(7):
            if row[i] != row[i]:
                row[i] =  0
        row[0] = int(row[0]) #PClass
        #Sex
        if row[1] == 'male':
            row[1] = 1
        if row[1] == 'female':
            row[1] = -1
        row[2] = float(row[2]) #Age
        row[3] = int(row[3]) #SibSp
        row[4] = int(row[4]) #Parch
        row[5] = float(row[5]) #Fare
        #Embarked
        embarked = [0, 'S', 'C', 'Q']
        row[6] = embarked.index(row[6])    
    return evidence
evidence = standardize(evidence)

#fit models
model1.fit(evidence, labels)
model2.fit(evidence, labels)
model3.fit(evidence, labels)
model4.fit(evidence, labels)

#compile test evidence
data2 = pd.read_csv('test.csv').drop(columns_to_be_removed, axis = 'columns')
testEvidence = data2.iloc[:, 1:].values.tolist()
testEvidence = standardize(testEvidence)

#make predictions using each model
predictions1 = model1.predict(testEvidence)
predictions2 = model2.predict(testEvidence)
predictions3 = model3.predict(testEvidence)
predictions4 = model4.predict(testEvidence)

#combine models for final prediction
finalPredictions = []
for i in range(len(predictions1)):
    predictionList = [predictions1[i], predictions2[i], predictions3[i], predictions4[i]]
    finalPredictions.append(statistics.mode(predictionList))

#list of passengers
passengers = data2.iloc[:, 0].values.tolist()

#create csv file
headerList = ['PassengerID', 'Survived']
with open('result.csv', 'w') as f:   
    dw = csv.DictWriter(f, delimiter=',', 
                        fieldnames=headerList)
    dw.writeheader()
    writer = csv.writer(f)
    writer.writerows(zip(passengers, finalPredictions))

'''
The code below was used initially to check how accurate each model was. 
Though K-nearest neighbors gave the highest accuracy at >90%, the combined model still outperformed
that of using K-nearest neighbors alone.

#check model accuracies
X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)
predictions1 = model1.predict(X_testing)
predictions2 = model2.predict(X_testing)
predictions3 = model3.predict(X_testing)
predictions4 = model4.predict(X_testing)

correct1 = (y_testing == predictions1).sum()
incorrect1 = (y_testing != predictions1).sum()
total1 = len(predictions1)
correct2 = (y_testing == predictions2).sum()
incorrect2 = (y_testing != predictions2).sum()
total2 = len(predictions2)
correct3 = (y_testing == predictions3).sum()
incorrect3 = (y_testing != predictions3).sum()
total3 = len(predictions3)
correct4 = (y_testing == predictions4).sum()
incorrect4 = (y_testing != predictions4).sum()
total4 = len(predictions4)

print(f"Results for model {type(model1).__name__}")
print(f"Correct: {correct1}")
print(f"Incorrect: {incorrect1}")
print(f"Accuracy: {100 * correct1 / total1:.2f}%")
print(" ")
print(f"Results for model {type(model2).__name__}")
print(f"Correct: {correct2}")
print(f"Incorrect: {incorrect2}")
print(f"Accuracy: {100 * correct2 / total2:.2f}%")
print(" ")
print(f"Results for model {type(model3).__name__}")
print(f"Correct: {correct3}")
print(f"Incorrect: {incorrect3}")
print(f"Accuracy: {100 * correct3 / total3:.2f}%")
print(" ")
print(f"Results for model {type(model4).__name__}")
print(f"Correct: {correct4}")
print(f"Incorrect: {incorrect4}")
print(f"Accuracy: {100 * correct4 / total4:.2f}%")
'''

MAX=20
from classifyDT import *
from classifyRF import *
from classifySVM import *
from classifyNN import *
from classifyLR import *

accuracy=0
for x in range(MAX):
    accuracy = accuracy + SVM()
accuracy/=MAX
print("SVM = ",accuracy)

accuracy=0
for x in range(MAX):
    accuracy = accuracy + DT()
accuracy/=MAX
print("Decision Tree = ",accuracy)

accuracy=0
for x in range(MAX):
    accuracy = accuracy + RF()
accuracy/=MAX
print("Random Forest = ",accuracy)

accuracy=0
for x in range(MAX):
    accuracy = accuracy + NN()
accuracy/=MAX
print("Nearest Neighbour = ",accuracy)

accuracy=0
for x in range(MAX):
    accuracy = accuracy + LR()
accuracy/=MAX
print("Linear Regression = ",accuracy)

# print("I conclude that SVM can work perfectly for smaller training sets while Random Forest"
# " and Decision Tree do not")

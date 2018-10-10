
# Dependencies
import tensorflow as tf
import pandas as pd
import numpy as np
import math
#import seaborn
#import matplotlib.pyplot as plt
import random
import csv





DecayRate = 0.25
InvestigationRate = 90

class RL_Record(object):
    def __init__(data):
        data.position = []
        data.error = []
        data.count = 0;
        
    def AddEntry(data, position, error):
        data.position.append(position)
        data.error.append(error)
        data.count = len(data.error) 
        if data.count > 10:
            deletepos = np.argmin(data.error)
            del data.error[deletepos]
            del data.position[deletepos]
            data.count = data.count - 1

    



test = RL_Record()


# Loading the dataset
#dataset =  pd.read_csv('iris_dataset.csv')
dataset = pd.read_csv('irisdatanormalised.csv')
#TargetData = pd.read_csv('targetdata.csv')
##log = pd.read_csv('file_path.csv')

#seaborn.pairplot(dataset)
#plt.show()

##load data into system and locate 
Learning = True


setosaMatch = np.array([0.48,0.24],dtype='float32')
versicolorMatch = np.array([0.8,0.75],dtype='float32')
virginicaMatch = np.array([0.2,0.77],dtype='float32')
TargetData = []
for i in range(0, len(dataset)):
    if dataset.iloc[i,4] == 'Iris-setosa':
        TargetData.append(setosaMatch)
    if dataset.iloc[i,4] == 'Iris-versicolor':
        TargetData.append(versicolorMatch)
    if dataset.iloc[i,4] == 'Iris-virginica':
        TargetData.append(virginicaMatch)
               

#y = TargetData
X = dataset

X.drop(X.columns[[4]], axis=1,  inplace=True)
#X.drop('Species', axis = 1)
X = np.array(X, dtype='float32')
y = np.array(TargetData, dtype='float32')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X, y, test_size=0.2, shuffle=False)
from sklearn.utils import shuffle
X_train,y_train = shuffle(X_train, y_train)






# Session
#sess =  tf.Session().as_default()

sess = tf.Session()

# Interval / Epochs
epoch = 80
learncount = 0
learnrate = 0.01
learnrateReset = 0.01
Correct = 0
setosa=0
setosaCorrect=0
versicolor=0
versicolorCorrect=0
virginica=0
virginicaCorrect=0

RL = False


# Initialize placeholders
X_data = tf.placeholder(shape=[None, 4], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 2], dtype=tf.float32)

# Input neurons : 4
# Hidden neurons : 8
# Output neurons : 3
hidden_layer_nodes = 50

# Create variables for Neural Network layers
w1 = tf.Variable(tf.random_normal(shape=[4,hidden_layer_nodes])) # Inputs -> Hidden Layer
b1 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # First Bias
w2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,hidden_layer_nodes])) # Hidden layer -> Outputs
b2 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes]))   # Second Bias
w3 = tf.Variable(tf.random_normal(shape=[hidden_layer_nodes,2])) # Hidden layer -> Outputs
b3 = tf.Variable(tf.random_normal(shape=[2]))   # Second Bias

# Operations
def NewLearnRate (learnrate):
    return  math.fabs(learnrate - (learnrate * DecayRate));

def InvestRate(InvestigationRate):
    return math.fabs(InvestigationRate + (InvestigationRate/10 * DecayRate));

def errorrate(a,b):
    return math.sqrt( math.pow(a[1] - b[1], 2) + math.pow(a[0] - b[0], 2))



hidden_output = tf.add(tf.matmul(X_data, w1), b1) 
hidden_output = tf.sigmoid(hidden_output)
hidden_output2 = tf.add(tf.matmul(hidden_output, w2), b2)
hidden_output2 = tf.sigmoid(hidden_output2)
final_output = tf.add(tf.matmul(hidden_output2, w3), b3)
final_output = tf.sigmoid(final_output)


# Cost Function
#loss = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(final_output), axis=0))

# Optimizer
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
diff = tf.subtract(final_output, y_target)


#loss = tf.pow(final_output,y)


cost = tf.multiply(diff, diff)
step = tf.train.GradientDescentOptimizer(learnrate).minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()


sess.run(init)
sess.as_default()
# Training
with open('dataoutput.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    Count = len(X_train)
    print('Training the model...')
    for e in range(1, (epoch + 1)):
        TotalError = 0;

        if (RL==False):
            sess.run(step, feed_dict={X_data: X_train, y_target: y_train})
            
            for i in range(0, (Count)):
                prediction = sess.run(final_output[0], feed_dict={X_data: X_train[i:], y_target: y_train[i:]})
                mytarget = sess.run(y_target[0],feed_dict={X_data: X_train[i:], y_target: y_train[i:]})
                error = errorrate(prediction,mytarget)
                TotalError = TotalError + error
                learncount+=1

                
                correctly_Found = False
       
                if np.array_equal(mytarget,setosaMatch):
                    #log = np.append(log,[prediction,mytarget,error])
                    setosa = setosa + 1
                    if error < 0.30:
                        Correct+=1
                        correctly_Found = True
                        setosaCorrect= setosaCorrect+1


                if np.array_equal(mytarget,versicolorMatch):
                    versicolor = versicolor + 1
                    if error < 0.30:
                        Correct+=1
                        correctly_Found = True
                        versicolorCorrect= versicolorCorrect+1

                if np.array_equal(mytarget,virginicaMatch):
                    virginica = virginica + 1
                    if error < 0.30:
                        Correct+=1
                        correctly_Found = True
                        virginicaCorrect= virginicaCorrect+1
            

                
                writer.writerow([prediction[0],prediction[1],mytarget[0],mytarget[1],error,learncount,'NORL',correctly_Found,learnrate,Learning,InvestigationRate, test.count])
#####normalsystem end



        else:
            for i in range(0, (Count)):

              
                if (Learning):
                            
                            sess.run(step, feed_dict={X_data: X_train[i:], y_target: y_train[i:]})
                            learncount +=1
                

                prediction = sess.run(final_output[0], feed_dict={X_data: X_train[i:], y_target: y_train[i:]})
                mytarget = sess.run(y_target[0],feed_dict={X_data: X_train[i:], y_target: y_train[i:]})
                error = errorrate(prediction,mytarget)
                TotalError = TotalError + error
           
            
                test.AddEntry(i,error)
            #    if (test.count > 10):
            #        minid = np.argmin(test.error)
            #        del test[minid]


      
                correctly_Found = False
       
                if np.array_equal(mytarget,setosaMatch):
                    #log = np.append(log,[prediction,mytarget,error])
                    setosa = setosa + 1
                    if error < 0.30:
                        Correct+=1
                        correctly_Found = True
                        learnrate = NewLearnRate(learnrate)
                        InvestigationRate = InvestRate(InvestigationRate)
                        setosaCorrect= setosaCorrect+1


                if np.array_equal(mytarget,versicolorMatch):
                    versicolor = versicolor + 1
                    if error < 0.30:
                        Correct+=1
                        correctly_Found = True
                        learnrate = NewLearnRate(learnrate)
                        InvestigationRate = InvestRate(InvestigationRate)
                        versicolorCorrect= versicolorCorrect+1

                if np.array_equal(mytarget,virginicaMatch):
                    virginica = virginica + 1
                    if error < 0.30:
                        Correct+=1
                        correctly_Found = True
                        learnrate = NewLearnRate(learnrate)
                        InvestigationRate = InvestRate(InvestigationRate)
                        virginicaCorrect= virginicaCorrect+1
            
                if (correctly_Found==False):
                    learnrate = learnrateReset
                    InvestigationRate = 90

                else:
                    if (InvestigationRate > 99.5):
                        Learning = False
                        InvestigationRate = 100
                        learnrate = 0
                    
                    else:
                        Learning = True

                writer.writerow([prediction[0],prediction[1],mytarget[0],mytarget[1],error,learncount,'Normal',correctly_Found,learnrate,Learning,InvestigationRate, test.count])

        #################################################################
        #### reluctan learning element
                if test.count > 2 and Learning:
            

                    maxid = np.argmax(test.error)
                    top =  test.position[maxid]
                    prediction = sess.run(final_output[0], feed_dict={X_data: X_train[top:], y_target: y_train[top:]})
                    mytarget = sess.run(y_target[0],feed_dict={X_data: X_train[top:], y_target: y_train[top:]})
                    TargetError = errorrate(prediction,mytarget)
                    TargetError = TargetError *0.98
                    stopcount = 10
                    while stopcount > 0:
                        sess.run(step, feed_dict={X_data: X_train[top:], y_target: y_train[top:]})
                        learncount = learncount + 1
                        prediction = sess.run(final_output[0], feed_dict={X_data: X_train[top:], y_target: y_train[top:]})
                        mytarget = sess.run(y_target[0],feed_dict={X_data: X_train[top:], y_target: y_train[top:]})
                        error = errorrate(prediction,mytarget)
                        test.error[maxid] = error
                        stopcount -=1
                        writer.writerow([prediction[0],prediction[1],mytarget[0],mytarget[1],error,learncount,'Normal',correctly_Found,learnrate,Learning,InvestigationRate, test.count])

                        if stopcount < 0 or error < TargetError:
                            stopcount = -1
    
    


            # Prediction

        print ('Total Average error for epoch = ', TotalError/learncount)

        print ('Learn Count = ', learncount)
        print (' Correct = ', Correct)
        print ('Correct %' , Correct/learncount*100)
        print ('setosa Match = ', setosa)
        print ('Setos correct', setosaCorrect/setosa*100)
        print ('virginica Match = ', virginica)
        print ('virginica correct', virginicaCorrect/virginica*100)
        print ('versicolor Match = ', versicolor)
        print ('versicolor correct', versicolorCorrect/versicolor*100)
        setosa=0
        setosaCorrect=0
        versicolor=0
        versicolorCorrect=0
        virginica=0
        virginicaCorrect=0

csvFile.close()

print ('Testing Phase')

with open('test.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    setosa=0
    setosaCorrect=0
    versicolor=0
    versicolorCorrect=0
    virginica=0
    virginicaCorrect=0
    size = len(y_test)
    for i in range(0, size):
            
                prediction = sess.run(final_output[0], feed_dict={X_data: X_test[i:], y_target: y_test[i:]})
                mytarget = sess.run(y_target[0],feed_dict={X_data: X_test[i:], y_target: y_test[i:]})
                error = errorrate(prediction,mytarget)
                TotalError = TotalError + error
                correctly_Found = False
                if np.array_equal(mytarget,setosaMatch):
                    #log = np.append(log,[prediction,mytarget,error])
                    setosa = setosa + 1
                    if error < 0.30:
                        Correct+=1
                        correctly_Found = True
                        setosaCorrect= setosaCorrect+1


                if np.array_equal(mytarget,versicolorMatch):
                    versicolor = versicolor + 1
                    if error < 0.30:
                        Correct+=1
                        correctly_Found = True
                        versicolorCorrect= versicolorCorrect+1

                if np.array_equal(mytarget,virginicaMatch):
                    virginica = virginica + 1
                    if error < 0.30:
                        Correct+=1
                        correctly_Found = True
                        virginicaCorrect= virginicaCorrect+1
            
           

                writer.writerow([prediction[0],prediction[1],mytarget[0],mytarget[1],error,'TESTING',correctly_Found])
    writer.writerow ('Total Average error for test  = ', TotalError/size)
    writer.writerow (' Correct = ', Correct)
    writer.writerow ('Correct %' , Correct/learncount*100)
    writer.writerow('setosa Match = ', setosa)
    writer.writerow('Setos correct', setosaCorrect/setosa*100)
    writerow('virginica Match = ', virginica)
    writer.writerow('virginica correct', virginicaCorrect/virginica*100)
    writer.writerow('versicolor Match = ', versicolor)
    writer.writerow('versicolor correct', versicolorCorrect/versicolor*100)

print ('Total Average error for test  = ', TotalError/size)
print (' Correct = ', Correct)
print ('Correct %' , Correct/learncount*100)
print ('setosa Match = ', setosa)
print ('Setos correct', setosaCorrect/setosa*100)
print ('virginica Match = ', virginica)
print ('virginica correct', virginicaCorrect/virginica*100)
print ('versicolor Match = ', versicolor)
print ('versicolor correct', versicolorCorrect/versicolor*100)
    #df = pd.DataFrame(log)
    #df.to_csv("file_path.csv",sep=',')

csvFile.close()

#seaborn.pairplot(df)
#plt.show()
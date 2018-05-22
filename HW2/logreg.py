import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

def getTrainingTestValidationData():
    TrainX=[]
    TestX=[]
    ValidationX=[]
    Trainy=[]
    Testy=[]
    Validationy=[]
    X = []
    y = []
    
    f=open(r'C:/Users/kazIm/Desktop/Python/mlhw2/HW2 Data/BreastCancer.csv','r')
    reader = csv.reader(f)

    for row in reader:
        X.append(row[1:])
        y.append(1 if (row[0]=='1') else 0)

    X = preprocessing.scale(X)#normalize

    TrainX,TestX,ValidationX,Trainy,Testy,Validationy = X[0:369],X[369:469],X[469:569],y[0:369],y[369:469],y[469:569]

    return np.asarray(TrainX,dtype='float64'),np.asarray(TestX,dtype='float64'),np.asarray(ValidationX,dtype='float64'),np.asarray(Trainy,dtype='float64'),np.asarray(Testy,dtype='float64'),np.asarray(Validationy,dtype='float64')

def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))

def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
    return ll

def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))        
    weights = np.zeros(features.shape[1])
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient
        
    return weights

def tunedParam(TrainX,ValidationX,Trainy,Validationy):
    acc=[]
    max_iters=[]
    learning_rates=[]
    num_steps = 100
    learning_rate = 1

    for i in range(11):
        weights = logistic_regression( TrainX, Trainy, num_steps, learning_rate, False)
        final_scores = np.dot(ValidationX, weights)
        preds = np.round(sigmoid(final_scores))
        acc.append("%.2f" % accuracy_score(preds, Validationy))
        max_iters.append(num_steps)
        learning_rates.append(learning_rate)
        num_steps += 1000
        learning_rate *= 1e-1
    index = acc.index(max(acc)) 
    print('---------Parameters tested for tuning----------')
    for i in range(len(acc)):
        print('max iteration:', max_iters[i],'learning rate:', learning_rates[i])
    
    return max_iters[index],learning_rates[index]

def traintestvalidationacc(max_iter,learning_rate,TrainX,TestX,ValidationX,Trainy,Testy,Validationy):
    print('Changing number of iteretions')
    acctrain=[]
    acctest=[]
    accval=[]
    max_iters=[]
    max_iter2 = 100
    for i in range(10):
        weights = logistic_regression( TrainX, Trainy, max_iter2, learning_rate, False)
        max_iters.append(max_iter2)

        final_scores = np.dot(ValidationX, weights)
        preds = np.round(sigmoid(final_scores))
        accval.append("%.2f" % accuracy_score(preds, Validationy))
        
        final_scores = np.dot(TrainX, weights)
        preds = np.round(sigmoid(final_scores))
        acctrain.append("%.2f" % accuracy_score(preds, Trainy))

        final_scores = np.dot(TestX, weights)
        preds = np.round(sigmoid(final_scores))
        acctest.append("%.2f" % accuracy_score(preds, Testy))
        
        max_iter2 += 1000
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(332)
    plt.plot(max_iters, acctrain)
    plt.xlabel('Max iteration', fontsize=14)
    plt.ylabel('Accuracy Train', fontsize=14) 
    plt.subplot(335)
    plt.plot(max_iters, accval)
    plt.xlabel('Max iteration', fontsize=14)
    plt.ylabel('Accuracy Validation', fontsize=14) 
    plt.subplot(338)
    plt.plot(max_iters, acctest)
    plt.xlabel('Max iteration', fontsize=14)
    plt.ylabel('Accuracy Test', fontsize=14) 
    plt.show()
    

    #####################################
    print('Changing learning rate')
    acctrain=[]
    acctest=[]
    accval=[]
    learning_rates=[]
    learning_rate2=1000
    for i in range(10):
        weights = logistic_regression( TrainX, Trainy, max_iter, learning_rate2, False)
        learning_rates.append(learning_rate2)

        final_scores = np.dot(ValidationX, weights)
        preds = np.round(sigmoid(final_scores))
        accval.append("%.2f" % accuracy_score(preds, Validationy))
        
        final_scores = np.dot(TrainX, weights)
        preds = np.round(sigmoid(final_scores))
        acctrain.append("%.2f" % accuracy_score(preds, Trainy))

        final_scores = np.dot(TestX, weights)
        preds = np.round(sigmoid(final_scores))
        acctest.append("%.2f" % accuracy_score(preds, Testy))
        
        learning_rate2 *= 1e-1
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(332)
    plt.semilogx(learning_rates, acctrain)
    plt.xlabel('Learning rates', fontsize=14)
    plt.ylabel('Accuracy Train', fontsize=14)
    plt.subplot(335)
    plt.semilogx(learning_rates, accval)
    plt.xlabel('Learning rates', fontsize=14)
    plt.ylabel('Accuracy Validation', fontsize=14) 
    plt.subplot(338)
    plt.semilogx(learning_rates, acctest)
    plt.xlabel('Learning rates', fontsize=14)
    plt.ylabel('Accuracy Test', fontsize=14) 
    plt.show()
    
        
def main():
    TrainX,TestX,ValidationX,Trainy,Testy,Validationy=getTrainingTestValidationData()
    #weights = logistic_regression( TrainX, Trainy,num_steps = 100000, learning_rate = 5e-5, add_intercept=False)
    #data_with_intercept = np.hstack((np.ones((ValidationX.shape[0], 1)),ValidationX))
    #final_scores = np.dot(data_with_intercept, weights)
    #final_scores = np.dot(ValidationX, weights)
    #preds = np.round(sigmoid(final_scores))
    #print('Accuracy : {0}'.format((preds == Validationy).sum().astype(float) / len(preds)))
    print("-------------------------------")
    print("         Question 2.1")
    print("-------------------------------")
    max_iter,learning_rate = tunedParam(TrainX,ValidationX,Trainy,Validationy)
    weights = logistic_regression( TrainX,Trainy,max_iter,learning_rate,False)
    final_scores = np.dot(TestX, weights)
    preds = np.round(sigmoid(final_scores))
    accTestData = "%.2f" % accuracy_score(preds,Testy)
    print("\nTuned parameters: max iteration:", max_iter,"learning rate:", learning_rate)
    print("Accuracy on test data using tuned parameters:", accTestData)
    print("-------------------------------")
    print("         Question 2.2")
    print("-------------------------------")
    traintestvalidationacc(max_iter,learning_rate,TrainX,TestX,ValidationX,Trainy,Testy,Validationy)
    print("-------------------------------")
    print("         Question 3")
    print("-------------------------------")

    #Question 3 

def crossvalidation(k=5):
    TrainX=[]
    TestX=[]
    ValidationX=[]
    Validationy=[]
    Trainy=[]
    Testy=[]
    acc=[]
    X = []
    y = []
    max_iters=np.arange(100,10100,1000)
    learning_rates=np.logspace(-7, 2, num=10)
    
    f=open(r'C:/Users/kazIm/Desktop/Python/mlhw2/HW2 Data/syntheticdataset.csv','r')
    reader = csv.reader(f)

    for row in reader:
        X.append(row[1:])
        y.append(1 if (row[0]=='1') else 0)

    # X = preprocessing.scale(X)#normalize
    k=5
    foldsize = int(len(X)/k)

    for iter in max_iters:
        for rate in learning_rates:
            for i in range(k):
                TrainX=[]
                TestX=[]
                ValidationX=[]
                Validationy=[]
                for j in range(len(X)):
                    if j % k == i:
                        ValidationX.append(X[j])
                        Validationy.append(y[j])
                    else:
                        TrainX.append(X[j])
                        Trainy.append(y[j])
               
                weights = logistic_regression( np.asarray(TrainX,dtype='float64'),np.asarray(Trainy,dtype='float64'),iter,rate,False)
                final_scores = np.dot(ValidationX, weights)
                preds = np.round(sigmoid(final_scores))
                accTestData = "%.2f" % accuracy_score(preds,Validationy)
                acc.append(accTestData)
               

    TrainX,TestX,ValidationX,Trainy,Testy,Validationy = X[0:369],X[369:469],X[469:569],y[0:369],y[369:469],y[469:569]

    return np.asarray(TrainX,dtype='float64'),np.asarray(TestX,dtype='float64'),np.asarray(ValidationX,dtype='float64'),np.asarray(Trainy,dtype='float64'),np.asarray(Testy,dtype='float64'),np.asarray(Validationy,dtype='float64')



if __name__ == "__main__":
    main()

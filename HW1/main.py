import numpy as np
import matplotlib.pyplot as plt
import math
import sys

class NaiveBayes(object):
    '''
    A Naive Bayes classifier application
    '''
    def __init__(self):
        #trainset data
        self.student, self.faculty, self.X1, self.y1 = [],[],[],[]
        #parameters to learn
        self.Tj0, self.Tj1 = [],[]
        self.sumT0, self.sumT1, self.N1, self.N = 0,0,0,0
        self.πY1 = 0.0
        self.θjy0, self.θjy1 = [],[]
        #test set data
        self.y_predicted, self.X2, self.y2 = [],[],[]

    def traindata(self, path_to_data):
    	'''
    	reading the data from the path and learning the parameters
    	'''

        with open(path_to_data) as data:
            for line in data:
                seperated = line.split(" ")
                if seperated[0] ==  'student':
                    self.student.append(seperated[1::])#dont put the true labels
                    self.X1.append(seperated[1::])
                    self.y1.append(seperated[0])
                else:
                    self.faculty.append(seperated[1::])
                    self.X1.append(seperated[1::])
                    self.y1.append(seperated[0])

        S = np.asanyarray(self.student).astype(np.int)#convert to numpy matrix
        #print('S dimensionality',S.shape)
        F = np.asanyarray(self.faculty).astype(np.int)
        #print('F dimensionality',F.shape)

        # check the shapes of X and y
        X = np.asarray(self.X1)
        y = np.asarray(self.y1)
        #print('X dimensionality', X.shape)
        #print('y dimensionality', y.shape)

        #learning parameters
        self.Tj0 = np.sum(F, axis=0)  # Compute sum of each column to learn Tj0
        self.Tj1 = np.sum(S, axis=0)  # Compute sum of each column to learn Tj1
        self.sumT0 = np.sum(self.Tj0)      # Compute sum of Tj0
        self.sumT1 = np.sum(self.Tj1)      # Compute sum of Tj1
        self.N1 = S.shape[0]         # Number student website in training set
        self.N = S.shape[0] + F.shape[0]   # Number total website in training set
        self.πY1 = self.N1/self.N     #estimated the probability that any particular document will be a student webpage.
        self.θjy0 = np.divide(self.Tj0 , self.sumT0 )    #P(Xj|Y = 0)
        self.θjy1 = np.divide(self.Tj1 , self.sumT1 )    #P(Xj|Y = 1)
    
    def MNB(self, path_to_data):
    	'''
    	predicting the data in the path using Multinomial Naive Base
    	'''
        with open(path_to_data) as data:
            for line in data:
                seperated = line.split(" ")
                self.X2.append(seperated[1::])
                self.y2.append(seperated[0])

        X = np.asarray(self.X2)
        y = np.asarray(self.y2)

        Twj = np.asanyarray(X).astype(np.int)

        prob_y0 = np.zeros(y.shape)#probabilities estimated for all documents being class faculty 
        prob_y1 = np.zeros(y.shape)

        for i in range(Twj.shape[0]):#getting probabilities
            for j in range(Twj.shape[1]):
                if  self.θjy0[j] == 0:
                    if Twj[i][j] == 0:
                        pass
                    else:
                        prob_y0[i] += -math.pow(10,100)#handling log(0) case
                else:
                    prob_y0[i] += Twj[i][j]*math.log(self.θjy0[j])
            prob_y0[i] += math.log(1-self.πY1)

        for i in range(Twj.shape[0]):
            for j in range(Twj.shape[1]):
                if  self.θjy1[j] == 0:
                    if Twj[i][j] == 0:
                        pass
                    else:
                        prob_y1[i] += -math.pow(10,100)
                else:
                    prob_y1[i] += Twj[i][j]*math.log(self.θjy1[j])
            prob_y1[i] += math.log(1-self.πY1)

        for i in range(Twj.shape[0]):#assigning classes
            if prob_y0[i] >= prob_y1[i]:
                self.y_predicted.append('faculty')
            else:
                self.y_predicted.append('student')

    def accuracy(self):
    	'''
    	comparing real labels with predicted labels to compute model accuracy
    	'''
        tru, fal = 0,0
        for i in range(len(self.y2)):
            if self.y2[i] == self.y_predicted[i]:
                tru += 1
            else:
                fal += 1
        return tru/(tru + fal)

    def MI(self):
    	'''
    	computing mutual information
    	'''
        mi = {}
        X = np.asarray(self.X1).astype(np.int)
        y = np.asarray(self.y1)
       
        for i in range(X.shape[1]):#1309 col
            a,n,n1,n0,n10,n11,n00,n01 = 0,0,0,0,0,0,0,0
            for j in range(X.shape[0]):#1000 row
                if X[j][i] == 0:
                    if y[j] == 'faculty':
                        n00 += 1
                    else:
                        n01 += 1
                else:
                    if y[j] == 'faculty':
                        n10 += 1
                    else:
                        n11 += 1
            n = n00+n01+n10+n11
            n1 = n10+n11
            n0 = n00+n01
            try: #handling log(0) cases
                a += ((n11/n)*math.log((n*n11)/(n1*n1),2))
            except (ValueError,ZeroDivisionError): 
                pass
            try: 
                a += ((n01/n)*math.log((n*n01)/(n0*n1),2))
            except (ValueError,ZeroDivisionError):
                pass
            try: 
                a += ((n10/n)*math.log((n*n10)/(n1*n0),2))
            except (ValueError,ZeroDivisionError):
                pass
            try: 
                a += ((n00/n)*math.log((n*n00)/(n0*n0),2))
            except (ValueError,ZeroDivisionError):
                pass
            
            mi[i] = a
            
            
        
        return mi
    
    def RFM(self,mi):
        '''
        recursive feature substraction using computed mutual information dictionary
        '''
        X = np.asarray(self.X1).astype(np.int)
        y = np.asarray(self.y1)
        numberoffeatures = [X.shape[1]]
        accuracy = [self.accuracy()]
        for i in range(len(mi)-1):
            #if(i == 30):
            #    break
            self.y_predicted = []
            X = np.delete(X, mi[i][0], 1 )#removing least informative feature
            for idx in mi:#fixing indecies after removing
                if idx[0]>mi[i][0]:
                    idx[0]-=1
            self.θjy0 = np.delete(self.θjy0, mi[i][0], 0)
            self.θjy1 = np.delete(self.θjy1, mi[i][0], 0)
            Twj = np.asanyarray(X).astype(np.int)

            prob_y0 = np.zeros(y.shape)
            prob_y1 = np.zeros(y.shape)

            for i in range(Twj.shape[0]):
                for j in range(Twj.shape[1]):
                    if  self.θjy0[j] == 0:
                        if Twj[i][j] == 0:
                            pass
                        else:
                            prob_y0[i] += -math.pow(10,100)
                    else:
                        prob_y0[i] += Twj[i][j]*math.log(self.θjy0[j])
                prob_y0[i] += math.log(1-self.πY1)

            for i in range(Twj.shape[0]):
                for j in range(Twj.shape[1]):
                    if  self.θjy1[j] == 0:
                        if Twj[i][j] == 0:
                            pass
                        else:
                            prob_y1[i] += -math.pow(10,100)
                    else:
                        prob_y1[i] += Twj[i][j]*math.log(self.θjy1[j])
                prob_y1[i] += math.log(1-self.πY1)

            for i in range(Twj.shape[0]):
                if prob_y0[i] >= prob_y1[i]:
                    self.y_predicted.append('faculty')
                else:
                    self.y_predicted.append('student')

            numberoffeatures.append(X.shape[1])#adding feature number and accuracy in an array
            accuracy.append(self.accuracy())
        return (numberoffeatures, accuracy)

def main(argv):
    trainpath = str(argv[:])+"/traindata.txt"
    testpath = str(argv[:])+"/testdata.txt"
    #print(argv[0])
    nb = NaiveBayes()
    # read train data
    print('Calculating accuracy on training data... \n')
    #print(trainpath)
    nb.traindata(trainpath)
    nb.MNB(trainpath)
    #print(nb.y_predicted)
    print("Accuracy of prediction is: ",str(nb.accuracy()*100) + "%\n")
    print('Confusion matrix:')
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(nb.y2, nb.y_predicted)
    print(cm,'\n')

    print('Calculating accuracy on test data... \n')
    nb2 = NaiveBayes()
    nb2.traindata(trainpath)
    nb2.MNB(testpath)
    #print(nb.y_predicted)
    print("Accuracy of prediction is: ",str(nb2.accuracy()*100) + "%\n")
    print('Confusion matrix:')
    print(confusion_matrix(nb2.y2, nb2.y_predicted))

    print('\nRanking features...')
    unsortedMI = nb.MI()
    import operator
    sorted_MI = sorted(unsortedMI.items(), key=operator.itemgetter(1))
    print('10 best feautures with indices: ')
    print((sorted_MI[::-1])[:10])
    print('plotting test-set accuracy/removed number of features...')
    (number_of_features, accuray) = nb.RFM(np.asarray(sorted_MI).astype(np.int))
    plt.plot(number_of_features,accuray) 
    plt.xlabel('number_of_features')
    plt.ylabel('accuray')
    plt.show()
    max_acc = accuray[0]
    max_ind = 0
    for i in range(len(accuray)):
        if accuray[i]>max_acc:
            max_acc = accuray[i]
            max_ind = i   	
    print('Max accuracy observed is:', max_acc)   
    print('Max accuracy is observed with', number_of_features[max_ind], 'features')     

    

if __name__ == '__main__':
    main(sys.argv[1])

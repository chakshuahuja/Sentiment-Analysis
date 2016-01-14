import nltk,sys,random
from sklearn import svm,tree, neighbors
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import GradientBoostingClassifier
from FeatureSelectors import UniqueTokenFrequencyFeatureSelector
#from FeatureSelectors import A
from FeatureSelectors import InformationGain, InformationGainWithStopWords, InformationGainWithSynonymGrouping,InformationGainWithStopWordsSynonymGrouping
def parseArff(filename):
    review_lines = open(filename).read().split('\n')
    reviews =  [','.join(r.split(',')[:-1])[1:-1].decode('string_escape').strip() for r in review_lines if r]
    labels = [r.split(',')[-1] for r in review_lines if r]
    if len(reviews) != len(labels):
        print 'Error Parsing Arff file, not all reviews are labeled'
        return None
    return reviews,labels

def dataToFeatureSetGen(data, selector):
    for i, (review, label) in enumerate(data):
        if i%10 == 0:
            sys.stdout.write('\rSelector:%i/%i' % (i+1,len(data)))
            sys.stdout.flush()
        yield (selector(review),label)
    print

def accuracy(clf, Xt, yt):
    predictions = clf.predict(Xt)
    TP, TN, FP, FN = 0,0,0,0
    for p, a  in zip (predictions, yt):
        if p == a and p:
            TP += 1
        elif p == a and not p:
            TN += 1
        elif p != a and p:
            FP += 1
        elif p != a and not p:
            FN += 1
    accuracy = (TP+TN) / float(TP+TN+FP+FN)
    precision = TP     / float(TP+FP)
    recall =    TP     / float(TP+FN)
    fmeasure = (2*precision*recall)/(precision+recall)
    return precision, recall, accuracy, fmeasure

def myFunc(test_data, train_data, k, files):
    selector = InformationGain(files[1]).selector
#    selector = AdjectivesOnly(files[1]).selector
#selector = InformationGainWithStopWords(files[1], files[2]).selector
    #selector = InformationGainWithSynonymGrouping(files[1]).selector
    #selector = InformationGainWithStopWordsSynonymGrouping(files[1], files[2]).selector
    print 'Training classifierer with', selector.im_class.__name__
#    clf = svm.SVC(kernel='linear')
 #   clf = tree.DecisionTreeClassifier() 
    #clf = GaussianNB()
#    clf = BernoulliNB()
    #clf = neighbors.KNeighborsClassifier(13, weights=['uniform','distance'][1])
#    clf = GradientBoostingClassifier()
    clf = MultinomialNB()
    X,y  = [],[]
    for features,label in dataToFeatureSetGen(train_data, selector):
        X.append(features.values())
        y.append(int(label=='pos'))
    clf.fit(X,y)
    Xt,yt = [],[]
    for features,label in dataToFeatureSetGen(test_data, selector):
        Xt.append(features.values())
        yt.append(int(label=='pos'))
    
    return accuracy(clf, Xt, yt)

def cross_fold(pos, neg, k, files, func):
    if len(pos) != len(neg):
        print 'Length of Positive and Negative not same'
        return None
    l = len(pos)
    ni = l/k #number of iterations
    print 'ni', ni
    accuracy_list = []
    P_list = []
    R_list = []
    A_list = []
    F_list = []

    for i in range(ni):
        si = i*k
        ei = (i+1)*k
        test_slice = slice(si, ei)
        train_slice1 = slice(0, si)
        train_slice2 = slice(ei, l)
        test_data = pos[test_slice] + neg[test_slice]
        train_data = pos[train_slice1]+pos[train_slice2]+neg[train_slice1]+neg[train_slice2]
        
        P,R,A,F = func(test_data, train_data, i, files)
        P_list.append(P)
        R_list.append(R)
        A_list.append(A)
        F_list.append(F)
    print 'Precision over all iterations', P_list
    print 'Recall over all iterations', R_list
    print 'Accuracy over all iterations', A_list
    print 'F-Measure over all iterations', F_list
    
    print 'Precision', sum(P_list)/ len(P_list)
    print 'Recall', sum(R_list)/ len(R_list)
    print 'Accuracy', sum(A_list)/ len(A_list)
    print 'F-Measure', sum(F_list)/ len(F_list)



def main(files):
    reviews, labels = parseArff(files[0])
    data = zip(reviews, labels)
#    random.shuffle(data)
    pos, neg = [d for d in data if d[1] == 'pos'], [d for d in data if d[1] == 'neg']
    #pos=pos[:500]
    #neg=neg[:500]
    print "Read %i reviews from %s. %i Positive, %i Negative" % (len(pos)+len(neg), sys.argv[1], len(pos), len(neg))
    cross_fold(pos, neg, 100, files, myFunc)

if __name__ == '__main__':
    main(sys.argv[1:])




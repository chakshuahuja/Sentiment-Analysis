import nltk
import re
import math
from nltk.util import ngrams
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer

def flatten(l):
    '''Converts [ [a,b], [c,d], [e,f,g] ] to [a,b,c,d,e,f,g]'''
    return [ s for sublist in l for s in sublist]

def uniqify(word_set):
    unique = list()
    unique_map = dict()
    words = word_set.copy()
    while words:
        a = words.pop()
        a_set = set([])
        i = len(unique)
        unique_map[a] = i
        words.difference_update(a_set)
        a_set.add(a)
        unique.append(a_set)
    return unique, unique_map
def sent_row(sent_tokens, uniqified_adj, uniqified_map):
    row = [0]* len(uniqified_adj)
    for tok in sent_tokens:
        if tok in uniqified_map:
            i = uniqified_map[tok]
            row[i] += 1
    return row

def review_tokenize(all_reviews):
    return re.split('\\n\',|,pos|,neg', all_reviews)

def tokenize(review):
    return re.split('[ ,\\n;!.?\']',review)

f = open('reviews.arff', 'r')
all_reviews = f.read()
sent_tokens = map(tokenize, review_tokenize(all_reviews))
tokens = flatten(sent_tokens)

unique_tokens, unique_tokens_map = uniqify(set(tokens))

matrix = [sent_row(s, unique_tokens, unique_tokens_map) for s in sent_tokens][:-1]

def getLabels():
    labels = all_reviews.split("\\n\',")
    return list([review[:3] for review in labels[1:]])

labels = getLabels()

for i in range(len(matrix)):
    matrix[i].append(labels[i])

def calculateEntropy(list_1):
    if len(list_1):
        pos_prob_1 = list_1.count('pos')/float(len(list_1))
        neg_prob_1 = 1 - pos_prob_1
        if pos_prob_1 and neg_prob_1:
            entropy_1 = -pos_prob_1*math.log(pos_prob_1,2) - neg_prob_1*math.log(neg_prob_1,2) 
        else:
            entropy_1 = 0
    else:
        entropy_1 = 0
    return entropy_1
    
def calculateInformationGain(token):
    col = unique_tokens_map[token]
    
    list_org = [row[-1] for row in matrix]
    entropy_before = calculateEntropy(list_org)
    #print 'Entropy_before : ', entropy_before
    
    list_1 = [matrix[i][-1] for i in range(len(matrix)) if matrix[i][col] != 0]
    entropy_1 = calculateEntropy(list_1)

    list_0 = [matrix[i][-1] for i in range(len(matrix)) if matrix[i][col] == 0]
    entropy_0 = calculateEntropy(list_0)
    
    entropy_after = len(list_1)/float(len(matrix))*entropy_1 + len(list_0)/float(len(matrix))*entropy_0
   # print 'Entropy_after : ', entropy_after
    
    information_gain = entropy_before - entropy_after
    return information_gain


def setInformationGain(unique_tokens):
    for token in unique_tokens:
        print list(token)[0], ' ', calculateInformationGain(list(token)[0])


setInformationGain(unique_tokens)

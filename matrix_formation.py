import nltk
from nltk.util import ngrams
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
def flatten(l):
    '''Converts [ [a,b], [c,d], [e,f,g] ] to [a,b,c,d,e,f,g]'''
    return [ s for sublist in l for s in sublist]
def synonyms(word):
    '''Returns synonyms of a word'''
    word = WordNetLemmatizer().lemmatize(word)
    return flatten(s.lemma_names() for s in wordnet.synsets(word))

def uniqify(word_set):
    '''
    Takes a word set and groups all synonyms together as list
    of frozensets. Returns this list as well as a map for find
    reference of the set.
    Eg: word_set = set(['crap','good','shit','nice','awkward'])
    Returns: (
    [set(['crap','shit']), set(['good','nice']), set(['awkward'])]   ,
    {crap:0, good:1, shit:0, nice:1, awkward:2}
    )
    '''
    unique = list()
    unique_map = dict()
    words = word_set.copy()
    while words:
        a = words.pop()
        a_set = set([])
        a_syn = synonyms(a)
        i = len(unique)
        unique_map[a] = i
        for b in words:
            if b in a_syn:
                a_set.add(b)
                unique_map[b] = i
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

f = open('check.txt', 'r')
all_reviews = f.read()
sent_tokens = map(nltk.word_tokenize, nltk.sent_tokenize(all_reviews))
tokens = flatten(sent_tokens)
#print "Unigrams :" , len(set(tokens))
#print "POS Tagging :", nltk.pos_tag(tokens)
pos_tagged = nltk.pos_tag(tokens)
adjectives = []
prepositions = []
nouns = []
adverbs = []
verbs = []
random = []
for token in pos_tagged:
    if token[1][:2] == 'JJ':
        adjectives.append(token[0])
    elif token[1] == 'IN':
        prepositions.append(token[0])
    elif token[1][:2] == 'NN':
        nouns.append(token[0])
    elif token[1][:2] == 'RB':
        adverbs.append(token[0])
    elif token[1][:2] == 'VB':
        verbs.append(token[0])
    else:
        random.append(token[0])

print "# of Adjectives : ", len(set(adjectives))
#print "Prepositions : ", set(prepositions)
#print "Nouns : ", set(nouns)
#print "Adverbs : ", set(adverbs)
#print "Verbs : ", set(verbs)
#print "Random : ", set(random)

unique_adj, unique_adj_map = uniqify(set(adjectives))
print "# of Unique Adjectives : ", len(unique_adj)
print unique_adj
print unique_adj_map

your_matrix = [sent_row(s, unique_adj, unique_adj_map) for s in sent_tokens]
for r in your_matrix:
    print r


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
tokenizer = RegexpTokenizer(r'\w+')

############ PHASE 1 ########################
class UniqueTokenFrequencyFeatureSelector:
    def __init__(self, reviews):
        tokenized = [tokenizer.tokenize(r) for r in reviews]
        tokens = set(self._flatten(tokenized))
        print 'UniqueTokenFrequencyFeatureSelector:TotalTokens:', len(tokens)
        self.tokens = tokens
       
    def _flatten(self,l):
        '''Converts [ [a,b], [c,d], [e,f,g] ] to [a,b,c,d,e,f,g]'''
        return [ s for sublist in l for s in sublist]

    def selector(self, review):
        tokens = self.tokens
        review_tokens = tokenizer.tokenize(review)
        features = dict(zip(tokens, [0]*len(tokens)))
        for tok in review_tokens:
            if tok in tokens:
                features[tok] = 1
        return features

#################### PHASE 2 ########################################

class InformationGain(UniqueTokenFrequencyFeatureSelector):
    def __init__(self, info_gain_file):
        f = open(info_gain_file,'r')
	threshold = 0.001
	line = [x.split()[-1] for x in f][:5000]#if float(x.split()[0]) > threshold]
	print 'Initially all words above threshold', threshold, len(line)
	self.tokens = set(line)
	
#class InformationGain(UniqueTokenFrequencyFeatureSelector):
 #   def __init__(self, info_gain_file):
  #      f = open(info_gain_file,'r')
   #     threshold = 0.001
    #    line = set(x.split()[0] for x in f if float(x.split()[1]) > threshold)
	#print 'Initially all words above threshold', threshold, len(line)
        #self.tokens = line

class InformationGainWithStopWords(InformationGain):
    def __init__(self, info_gain_file, stop_word_file):
        InformationGain.__init__(self, info_gain_file)
	f = open(stop_word_file,'r')
	stop_words = set(f.read().split('\n'))
	print '# of Stop words', len(stop_words)
	remaining_words = set(self.tokens).difference(stop_words)
        print 'After removing stop words', len(remaining_words)
        self.tokens = remaining_words


class InformationGainWithSynonymGrouping(InformationGain):
    def __init__(self, info_gain_file):
        InformationGain.__init__(self, info_gain_file)
        self.tokens_grouped, self.tokens = self._uniqify(self.tokens)
        self.tokens_grouped = map(frozenset,self.tokens_grouped)
        print 'After Groupin Synonyms:', len(self.tokens_grouped)

    def selector(self, review):
        tokens = self.tokens
        tokens_grouped = self.tokens_grouped
        review_tokens = tokenizer.tokenize(review)
        features = dict(zip(tokens_grouped, [0]*len(tokens_grouped)))
        for tok in review_tokens:
            if tok in tokens:
                grouped_token = tokens_grouped[tokens[tok]]
                features[grouped_token] = 1
        return features
    def _flatten(self,l):
        '''Converts [ [a,b], [c,d], [e,f,g] ] to [a,b,c,d,e,f,g]'''
        return [ s for sublist in l for s in sublist]

    def _synonyms(self,word):
        '''Returns synonyms of a word'''
        word = WordNetLemmatizer().lemmatize(word)
        return self._flatten(s.lemma_names() for s in wordnet.synsets(word))

    def _uniqify(self,word_set):
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
            a_syn = self._synonyms(a)
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

class InformationGainWithStopWordsSynonymGrouping(InformationGainWithSynonymGrouping):
    def __init__(self, info_gain_file, stop_word_file):
        InformationGain.__init__(self, info_gain_file)
	f = open(stop_word_file,'r')
	stop_words = set(f.read().split('\n'))
	print '# of Stop words', len(stop_words)
	remaining_words = set(self.tokens).difference(stop_words)
        print 'After removing stop words', len(remaining_words)
        self.tokens = remaining_words
        ############################################################
        self.tokens_grouped, self.tokens = self._uniqify(self.tokens)
        self.tokens_grouped = map(frozenset,self.tokens_grouped)
        print 'After Groupin Synonyms:', len(self.tokens_grouped)

########################### PHASE 3 ###################################

# Adjective : JJ , JJR (Comparative) JJS (Superlative)
class AdjectivesOnly(UniqueTokenFrequencyFeatureSelector):
    def __init__(self, tagged_file):
        pos_tagged =eval(open(tagged_file).read())
        adjectives = []
	adverbs = []
	for token in pos_tagged:
            if token[1][:2] == 'JJ':
                adjectives.append(token[0])
	    elif token[1][:2] == 'RB':
                adverbs.append(token[0])
	
	f = open('IG.txt', 'r')
        tokens_ig = [x.split()[-1] for x in f]
        tokens_ig = list(set(tokens_ig))
        line = set(tokens_ig[:2500]).intersection(adjectives)
        print 'Initially all words adjectives', len(line)
        self.tokens = line

# Adverbs : RB , RBS (Superlative), RBR (Comparative)
class AdverbsOnly(UniqueTokenFrequencyFeatureSelector):
    def __init__(self, tagged_file):
        pos_tagged =eval(open(tagged_file).read())
        adjectives = []
	adverbs = []
	for token in pos_tagged:
            if token[1][:2] == 'RB':
                adjectives.append(token[0])
	    elif token[1][:2] == 'RBS':
                adverbs.append(token[0])
	
	f = open('IG.txt', 'r')
        tokens_ig = [x.split()[-1] for x in f]
        tokens_ig = list(set(tokens_ig))
        line = set(tokens_ig[:2500]).intersection(adjectives)
        print 'Initially all words adverbs', len(line)
        self.tokens = line

# Nouns : NN , NNS (Singular) NNP (Plural)
class NounsOnly(UniqueTokenFrequencyFeatureSelector):
    def __init__(self, tagged_file):
        pos_tagged =eval(open(tagged_file).read())
        adjectives = []
	adverbs = []
	for token in pos_tagged:
            if token[1][:2] == 'NN':
                adjectives.append(token[0])
	    elif token[1][:2] == 'NNP':
                adverbs.append(token[0])
	
	f = open('GR.txt', 'r')
        tokens_ig = [x.split()[-1] for x in f]
        tokens_ig = list(set(tokens_ig))
        line = set(tokens_ig[:2500]).intersection(adjectives)
        print 'Initially all words nouns', len(line)
        self.tokens = line
class A(UniqueTokenFrequencyFeatureSelector):
    def __init__(self):
        f = open('ooutput2.txt','r')
        all_attr = f.read()
        attr = all_attr.split('\n')
        features = [' '.join(atr.split()).split()[2] for atr in attr[:-1]]
        self.tokens = features


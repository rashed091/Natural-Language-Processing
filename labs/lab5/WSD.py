"""
Word Sense Disambiguation using naive bayes 
"""

import nltk
import random
from nltk.corpus import senseval
from nltk.classify import accuracy, NaiveBayesClassifier, MaxentClassifier
from collections import defaultdict

"""
Set of Stopwords
"""
STOPWORDS = {'.', ',', '?', '"', '``', "''", "'", '--', '-', ':', ';', '(',
             ')', '$', '000', '1', '2', '10,' 'I', 'i', 'a', 'about', 'after', 'all', 
			 'also', 'an', 'any', 'are', 'as', 'at', 'and', 'be', 'being', 'because', 
			 'been', 'but', 'by', 'can', "'d", 'did', 'do', "don'", 'don', 'for', 
			 'from', 'had','has', 'have', 'he', 'her','him', 'his', 'how', 'if', 
			 'is', 'in', 'it', 'its', "'ll", "'m", 'me', 'more', 'my', 'n', 'no',
			 'not', 'of', 'on', 'one', 'or', "'re", "'s", "s", 'said', 'say', 'says',
			 'she', 'so', 'some', 'such', "'t", 'than', 'that', 'the', 'them', 'they',
			 'their', 'there', 'this', 'to', 'up', 'us', "'ve", 'was', 'we', 'were',
             'what', 'when', 'where', 'which', 'who', 'will', 'with', 'years', 'you','your'}
inst_cache = {}
"""
Extra Utility function to work with senseval corpus
They depend on the (non-obvious?) fact that although the field is called 'senses', 
there is always only 1, i.e. there is no residual ambiguity in the data as we 
have it, because this is the gold standard and disambiguation per the context 
has already been done.
"""
def instance2senses(word):
    """
	Return the list of possible senses for a word per senseval-2
    
    :param word: The word to look up
    :type word: str
    :return: list of senses
    :rtype: list(str)
    """
    return list(set(i.senses[0] for i in senseval.instances(word)))


def sense2instances(instances, sense):
    """
	Return a list of instances that have the given sense
    
    :param instances: corpus of sense-labelled instances
    :type instances: list(senseval.SensevalInstance)
    :param sense: The target sense
    :type sense: str
    :return: matching instances
    :rtype: list(senseval.SensevalInstance)
    """
    return [instance for instance in instances if instance.senses[0]==sense]


def context_features(instance, vocab, dist=3):
	"""
	Return a featureset dictionary of left/right context word features within a distance window
	of the sense-classified word of a senseval-2 instance, also a feature for the word and for
	its part of speech,
	for use by an NLTK classifier such as NaiveBayesClassifier or MaxentClassifier
	
	:param instance: sense-labelled instance to extract features from
	:type instance: senseval.SensevalInstance
	:param vocab: ignored in this case
	:type vocab: str
	:param dist: window size
	:type dist: int
	:return: feature dictionary
	:rtype: dict
	"""
	features = {}
	idx = instance.position
	con = instance.context
	for i in range(max(0, idx-dist), idx):
		j = idx-i
		features['left-context-word-{}({})'.format(j, con[i][0])] = True

	for i in range(idx+1, min(idx+dist+1, len(con))):
		j = i-idx
		features['right-context-word-{}({})'.format(j, con[i][0])] = True

	features['word'] = instance.word
	features['pos'] = con[1][1]
	return features

def word_features(instance, vocab, dist=3):
	"""
	Return a featureset for an NLTK classifier such as NaiveBayesClassifier or MaxentClassifier
	where every key returns False unless it occurs in the instance's context
	and in a specified vocabulary
	
	:param instance: sense-labelled instance to extract features from
	:type instance: senseval.SensevalInstance
	:param vocab: filter for context words that yield features
	:type vocab: list(str)
	:param dist: ignored in this case
	:type dist: int
	:return: feature dictionary
	:rtype: dict
	"""
	features = defaultdict(lambda:False)
	features['alwayson'] = True
	# Not all context items are (word,pos) pairs, for some reason some are just strings...
	for w in (e[0] for e in instance.context if isinstance(e,tuple)):
			if w in vocab:
				features[w] = True
	return features

def extract_vocab_frequency(instances, n=300):
	"""
	Construct a frequency distribution of the non-stopword context words
	in a collection of senseval-2 instances and return the top n entries, sorted
	
	:param instances: sense-labelled instances to extract from
	:type instance: list(senseval.SensevalInstance)
	:param n: number of items to return
	:type n: int
	:return: sorted list of at most n items from the frequency distribution
	:rtype: list(tuple(str,int))
	"""
	fd = nltk.FreqDist()
	for i in instances:
		(target, suffix) = i.word.split('-')
		words = (c[0] for c in i.context if not c[0] == target)
		for word in set(words) - STOPWORDS:
			fd[word] += 1
	return fd.most_common()[:n+1]

	
def extract_vocab(instances, n=300):
	"""
	Return the n most common non-stopword words appearing as context
	in a collection of semeval-2 instances
		
	:param instances: sense-labelled instances to extract from
	:type instance: list(senseval.SensevalInstance)
	:param stopwords: words to exclude from the result
	:type stopwords: iterable(string)
	:param n: number of words to return
	:type n: int
	:return: sorted list of at most n words
	:rtype: list(str)"""

	return [w for w,f in extract_vocab_frequency(instances,n)]

def WSDClasifier(trainer, 
                 word,
				 features,
				 stopwords=STOPWORDS, 
				 number=300,
				 distance=3,
				 log=False,
				 confusion_matrix=False):
	"""
	Build a classifier instance for the senseval2 senses of a word and applies it

	:param word: from senseval2 (we have 'hard.pos', 'interest.pos', 'line.pos' and 'serve.pos')
	:type string:
	:param features: selector to which feature set to use
	:type features: str (word, context)
	:param n: passed to extract_vocab when constructing the second argument to the feature set constructor
	:type int:
	:param dist: passed to the feature set constructor as 3rd argument
	:type int:
	:param log: if set to True outputs any errors into a file errors.txt
	:type bool:
	:param confusion_matrix: if set to True prints a confusion matrix
	:type bool:

	Calling this function splits the senseval data for the word into a training set and a test set (the way it does
	this is the same for each call of this function, because the argument to random.seed is specified,
	but removing this argument would make the training and testing sets different each time you build a classifier).

	It then trains the trainer on the training set to create a classifier that performs WSD on the word,
	using features (with number or distance where relevant).

	It then tests the classifier on the test set, and prints its accuracy on that set.

	If log==True, then the errors of the classifier over the test set are written to errors.txt.
	For each error four things are recorded: (i) the example number within the test data (this is simply the index of the
	example within the list test_data); (ii) the sentence that the target word appeared in, (iii) the
	(incorrect) derived label, and (iv) the gold label.

	If confusion_matrix==True, then calling this function prints out a confusion matrix, where each cell [i,j]
	indicates how often label j was predicted when the correct label was i (so the diagonal entries indicate labels
	that were correctly predicted).
	"""
	global inst_cache

	if word not in inst_cache:
		inst_cache[word] = [(i, i.senses[0]) for i in senseval.instances(word)]
		
	events = inst_cache[word][:]
	senses = list(set(l for (i, l) in events))
	instances = [i for (i, l) in events]
	vocab = extract_vocab(instances, number)
	print(' Senses: ' + ' '.join(senses))
	# Split the instances into a training and test set,
	#if N > len(events): N = len(events)
	N = len(events)
	random.seed(123456789) 
	random.shuffle(events)
	train_data = events[:int(0.8 * N)]
	test_data = events[int(0.8 * N):N]

	# Train classifier
	print('Training classifier...')
	classifier = trainer([(features(i, vocab, distance), label) for (i, label) in train_data])
	# Test classifier
	print('Testing classifier...')
	acc = accuracy(classifier, [(features(i, vocab, distance), label) for (i, label) in test_data] )
	print('Accuracy: {:6.4f}'.format(acc))

	if log:
		#write error file
		print('Writing errors to errors.txt')
		with open('errors.txt', 'w') as file:
			errors = []
			for (i, label) in test_data:
				guess = classifier.classify(features(i, vocab, distance))
				if guess != label:
					con =  i.context
					position = i.position
					item_number = str(test_data.index((i, label)))
					word_list=[cv[0] if isinstance(cv,tuple) else cv for cv in con]
					hard_highlighted = word_list[position].upper()
					word_list_highlighted = word_list[0:position] + [hard_highlighted] + word_list[position+1:]
					sentence = ' '.join(word_list_highlighted)
					errors.append([item_number, sentence, guess,label])
			file.write('There are {} errors'.format(len(errors)))
			file.write('----------------------------\n')
			for error in errors:
				idx = errors.index(error)+1
				num, snt, guess, label = error
				file.write('{}) example #: {} \n sentence: {}\n guess: {}\n label: {}\n'.format(idx, num, snt, guess, label))
					
	if confusion_matrix:
		gold = [label for (i, label) in test_data]
		derived = [classifier.classify(features(i,vocab)) for (i,label) in test_data]
		cm = nltk.ConfusionMatrix(gold,derived)
		print(cm)
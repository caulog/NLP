import sys
from collections import defaultdict
import math
import random
import os
import os.path

"""
COMS W4705 - Natural Language Processing - Fall 2024 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None):
    # reads file and represents each sentence (each line) as a list of tokens
    # to deal w unseen words/contexts replace w "UNK" token
    with open(corpusfile,'r') as corpus:
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  

def get_ngrams(sequence, n):
    ngrams = []

    # ensure n is greater than 0
    if n < 1:
        print("invalid n")
        return ngrams

    # add START and STOP to sequence
    # don't count start
    # start = ['START']
    if n > 1: start *= (n-1)
    stop = ['STOP']
    sequence = start + sequence + stop

    # create ngrams
    for i in range(len(sequence) - n+1):
        ngrams.append(tuple(sequence[i:i+n]))

    return ngrams


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):

        # count how many times each unigram, bigram, and trigram appears in the lexicon
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {}
        self.total = {}

        # for each line in the corpus file, create uni/bi/trigrams
        for line in corpus:
            unigram = get_ngrams(line, 1)
            bigram = get_ngrams(line, 2)
            trigram = get_ngrams(line, 3)

            # for each uni/bi/trigram update counts
            for gram in unigram:
                self.unigramcounts[gram] = self.unigramcounts.get(gram, 0) + 1
                # should be number of words ... including words ?
                self.total[0] = self.total.get(0,0) + 1
            for gram in bigram:
                self.bigramcounts[gram] = self.bigramcounts.get(gram, 0) + 1
            for gram in trigram:
                self.trigramcounts[gram] = self.trigramcounts.get(gram, 0) + 1

        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """

        # p(w|u,v) = count(u,v,w)/count(u,v)

        #print(trigram[0], trigram[1])
        #print(self.bigramcounts.get((trigram[0], trigram[1]), 0))
        count_uv = self.bigramcounts.get((trigram[0], trigram[1]), 0)
        count_uvw = self.trigramcounts.get((trigram[0], trigram[1], trigram[2]), 0)

        return count_uvw/count_uv
        #return 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        count_u = self.unigramcounts.get((bigram[0]), 0)
        count_uv = self.bigramcounts.get((bigram[0], bigram[1]), 0)

        return count_u/count_uv
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        total = self.total.get(0,0)
        count_u = self.unigramcounts.get(unigram, 0)

        return count_u/total

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """

        # only matters here because uni and bigram taken care of w UNK tokens
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return 0.0
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        return float("-inf")

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        return float("inf") 


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
        
        return 0.0

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # Testing for get_ngrams
    print(get_ngrams(["natural", "language", "processing"], 3))

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)


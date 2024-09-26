import sys
from collections import defaultdict
import math
import random
import os
import os.path
from itertools import count

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
    start = ['START']
    if n > 1: start *= (n-1)
    stop = ['STOP']
    sequence = start + sequence + stop

    # create ngrams
    for i in range(len(sequence) - n+1):
        ngrams.append(tuple(sequence[i:i+n]))

    # when testing w ungraded_test why do somewords get UNK... i guess i dont understand how that func works
    # print(ngrams)

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
        self.lines = {}

        # for each line in the corpus file
        lines = 0
        for line in corpus:
            # get unigrams then update dictionary counts and total tokens
            unigrams = get_ngrams(line, 1)
            for ngram in unigrams:
                self.unigramcounts[ngram] = self.unigramcounts.get(ngram, 0) + 1
                self.total[0] = self.total.get(0,0) + 1

            # get bigrams then update dictionary counts
            bigrams = get_ngrams(line, 2)
            for ngram in bigrams:
                self.bigramcounts[ngram] = self.bigramcounts.get(ngram, 0) + 1

            # get trigrams then update dictionary counts
            trigrams = get_ngrams(line, 3)
            for ngram in trigrams:
                self.trigramcounts[ngram] = self.trigramcounts.get(ngram, 0) + 1

            # count number of lines for START
            lines += 1

        # store total number of TOKENS excluding START and total number of lines in corpus
        self.total[0] = self.total.get(0, 0) - lines
        self.lines[0] = lines

        return

    def raw_trigram_probability(self,trigram):

        # p(w|u,v) = count(u,v,w)/count(u,v)
        count_uvw = self.trigramcounts.get((trigram[0], trigram[1], trigram[2]), 0)
        count_uv = self.bigramcounts.get((trigram[0], trigram[1]), 0)

        # for ['START', 'START', 'TOKEN'] trigrams
        if (trigram[0] == 'START') & (trigram[1] == 'START'):
            count_uv = self.lines[0]

        # for unseen context P = (1/V)
        if count_uv == 0:
            return 1/(len(self.unigramcounts) - 1)

        '''For testing'''
        #print((trigram[0], trigram[1], trigram[2]), count_uvw)
        #print((trigram[0], trigram[1]), count_uv)

        return count_uvw/count_uv

    def raw_bigram_probability(self, bigram):

        # p(v|u) = count(u,v)/count(u)
        count_uv = self.bigramcounts.get((bigram[0], bigram[1]), 0)
        count_u = self.unigramcounts.get((bigram[0],), 0)

        # for unseen context P = (1/V)
        if count_u == 0:
            return 1/(len(self.unigramcounts) - 1)

        '''For testing'''
        #print((bigram[0], bigram[1]), count_uv)
        #print((bigram[0],), count_u)

        return count_uv/count_u

    def raw_unigram_probability(self, unigram):

        # p(u) = count(u)/total
        total = self.total.get(0,0)
        count_u = self.unigramcounts.get((unigram[0],), 0)

        '''For testing'''
        #print(total)
        #print(unigram, count_u)

        return count_u/total

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """

        return result            

    def smoothed_trigram_probability(self, trigram):
        # only matters here because uni and bigram taken care of w UNK tokens
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        # p(w|u,v) = lambda1 * p_mle(w|u,v) + lambda2 * p_mle(w|v) + lambda3 + p_mle(w)
        lambda_uvw = lambda1 * self.raw_trigram_probability((trigram[0], trigram[1], trigram[2]))
        lambda_vw = lambda2 * self.raw_bigram_probability((trigram[1], trigram[2]))
        lambda_w = lambda3 * self.raw_unigram_probability((trigram[2],))

        ''' For testing'''
        #print((trigram[0], trigram[1], trigram[2]), lambda_uvw, '\n')
        #print((trigram[1], trigram[2]), lambda_vw, '\n')
        #print((trigram[2],), lambda_w, '\n')

        return lambda_uvw + lambda_vw + lambda_w
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        # Use the get_ngrams function to compute trigrams
        trigrams = get_ngrams(sentence, 3)

        # Use the smoothed_trigram_probability method to obtain probabilities
        probability = 0
        for ngram in trigrams:
            ngram_prob = self.smoothed_trigram_probability(ngram)
            # Convert each probability into logspace using math.log2
            # Sum log probabilities
            probability += math.log2(ngram_prob)

        return probability

    def perplexity(self, corpus):
        """
          COMPLETE THIS METHOD (PART 6)
          Returns the log probability of an entire sequence.
          """
        # l = 1/M SUM(i = 1-> m)(log2 prob(sentence_i))
        # m = number of sentences in test data
        # perplexity = sum the log probability for each sentence
        # and then divide by the total number of words tokens in the test file.
        # M = the total number of word tokens in the test file

        M = 0
        m_sum = 0
        for line in corpus:
            M += len(line)
            logP = self.sentence_logprob(line)
            m_sum += logP

        l = (1 / M) * m_sum

        return math.pow(2, -l)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
        # Create two trigram models
        # high scoring
        model1 = TrigramModel(training_file1)
        # low scoring
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       

        # We compute the perplexity of each language model on each essay.

        # The model with the lower perplexity determines the class of the essay.


        # All you have to do is compare the perplexities and the returns
        # the accuracy (correct predictions / total predictions).

        # reads in the test essays from each directory
        # Find high scores
        for f in os.listdir(testdir1):
            total += 1
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if (pp_high < pp_low):
                correct += 1

        # Find low scores
        for f in os.listdir(testdir2):
            total += 1
            pp_low = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp_high = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if (pp_high > pp_low):
                correct += 1
        
        return (correct/total)*100

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1])

    # Testing for get_ngrams
    #print(get_ngrams(["natural", "language", "processing"], 3))

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


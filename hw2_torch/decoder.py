import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):

        # Creates state instance: 0 words on the stack
        # Buffer contains all input words (indices)
        # Deps structure is empty
        state = State(range(1,len(words)))
        state.stack.append(0)

        # Ideal: only select  highest scoring transition and update the state accordingly --> select the highest scoring permitted transition
        # Unfortunately, possible that highest scoring transition is not possible.
        # arc-left or arc-right are not permitted the stack is empty.
        # Shifting the only word out of the buffer is also illegal, unless the stack is empty.
        # Finally, the root node must never be the target of a left-arc
        # TODO: Write the body of this loop for part 5

        # As long as the buffer is not empty
        while state.buffer:
            # Use feature extractor to obtain current state
            # State is numpy array and needs to be a tensor reshaped w correct dimensions for model
            curr_state = torch.tensor(self.extractor.get_input_representation(words, pos, state)).reshape(1,-1)
            # model(features) and retrieve a softmax actived vector of possible actions
            actions = self.model(curr_state)
            # Create a list of possible actions and sort it according to their output probability
            actions_array = actions.detach().numpy()
            probable_actions = np.sort(actions_array)

            next_transition = False
            i = 0
            while not next_transition:
                # to make sure out of bounds doesn't occur
                #try:
                #    try_transition = probable_actions[0][i]
                #except IndexError as e:
                #    break
                #else:
                #    transition = self.output_labels[i]
                #    print(i, "\t", transition)

                transition = self.output_labels[i]
                print(i, "\t", transition)

                i = i + 1
                #print(transition)

            # (make sure the largest probability comes first in the list)
          #pass # replace


  

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))

        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()

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

        # As long as the buffer is not empty
        while state.buffer:
            # Use feature extractor to obtain current state
            # State is numpy array and needs to be a tensor reshaped w correct dimensions for model
            curr_state = torch.tensor(self.extractor.get_input_representation(words, pos, state)).reshape(1,-1)
            # model(features) to retrieve a softmax actived vector of possible actions
            actions = self.model(curr_state)
            softmax_actions = torch.softmax(actions, dim=-1)
            # Create a list of possible actions and sort it according to their output probability
            softmax_actions_array = softmax_actions[0].detach().numpy()
            probable_indexes = np.argsort(softmax_actions_array)[::-1][:20]

            find_transition = True
            i = 0
            while find_transition:
                # get the index of the most probable action
                '''try_transition = probable_indexes[0][i]'''
                try_transition = probable_indexes[i]
                # get the output label of that transition
                transition = self.output_labels[try_transition]

                # Get transition type and label
                name = transition[0]
                label = transition[1]

                # Shifting the only word out of the buffer is also illegal, unless the stack is empty.
                # arc-left or arc-right are not permitted the stack is empty.
                # the root node must never be the target of a left-arc
                if(name == "shift") and (len(state.buffer) > 1 or (len(state.buffer) == 1 and len(state.stack) == 0)):
                    find_transition = False
                if(name == "left_arc") and (len(state.stack) > 0) and (state.stack[-1] != 0):
                    find_transition = False
                if(name == "right_arc") and (len(state.stack) > 0):
                    find_transition = False

                i = i + 1

            # do transition
            if name == "shift":
                state.shift()
            if name == "left_arc":
                state.left_arc(label)
            if name == "right_arc":
                state.right_arc(label)

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

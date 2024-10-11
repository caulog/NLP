import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, NLLLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor

class DependencyDataset(Dataset):

  # Instantiates the DependencyDataset class and loads the data from the
  # target_train.npy and input_train.npy files from part 2
  '''There was an error here ... originally "inputs_filename" in parameter and used input_filename as variable??'''
  def __init__(self, input_filename, output_filename):
    self.inputs = np.load(input_filename)
    self.outputs = np.load(output_filename)

  # Returns the total number of input/target pairs in the dataset
  def __len__(self): 
    return self.inputs.shape[0]

  # Returns the input/target pair with index k
  def __getitem__(self, k): 
    return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 

  def __init__(self, word_types, outputs):
    super(DependencyModel, self).__init__()
    # TODO: complete for part 3
    # Embedding layer with
    # num_embeddings = number of word_types: size of the dictionary of embeddings
    # embedding_dim = 128: size of each embedding vector
    #print(word_types)
    self.embedding_layer = torch.nn.Embedding(word_types, 128)
    self.hidden_layer = torch.nn.Linear(128, 128)
    self.output_layer = torch.nn.Linear(128, 91)

  def forward(self, inputs):
    embedding_tensor= self.embedding_layer(inputs).view(len(inputs), 786)
    hidden_tensor = torch.nn.functional.relu(self.hidden_layer(embedding_tensor))
    output_tensor = self.output_layer(hidden_tensor)
    print(output_tensor.shape)
    # TODO: complete for part 3
    #return torch.zeros(inputs.shape(0), 91)  # replace this line
    return output_tensor


def train(model, loader): 

  loss_function = NLLLoss(reduction='mean')

  LEARNING_RATE = 0.01 
  optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

  tr_loss = 0 
  tr_steps = 0

  # put model in training mode
  model.train()
 

  correct = 0 
  total =  0 
  for idx, batch in enumerate(loader):
 
    inputs, targets = batch
 
    predictions = model(torch.LongTensor(inputs))

    loss = loss_function(predictions, targets)
    tr_loss += loss.item()

    #print("Batch loss: ", loss.item()) # Helpful for debugging, maybe 

    tr_steps += 1
    
    if idx % 1000==0:
      curr_avg_loss = tr_loss / tr_steps
      print(f"Current average loss: {curr_avg_loss}")

    # To compute training accuracy for this epoch 
    correct += sum(torch.argmax(logits, dim=1) == torch.argmax(targets, dim=1))
    total += len(inputs)
      
    # Run the backward pass to update parameters 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


  epoch_loss = tr_loss / tr_steps
  acc = correct / total
  print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")


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


    model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))

    dataset = DependencyDataset(sys.argv[1], sys.argv[2])
    loader = DataLoader(dataset, batch_size = 16, shuffle = True)

    print("Done loading data")

    # Now train the model
    for i in range(5): 
      train(model, loader)


    torch.save(model.state_dict(), sys.argv[3]) 

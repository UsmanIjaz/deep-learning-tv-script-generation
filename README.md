

## TV Script Generation

In this project, we'll generate our own [Seinfeld](https://en.wikipedia.org/wiki/Seinfeld) TV scripts using RNNs.  We'll be 
using part of the [Seinfeld dataset](https://www.kaggle.com/thec03u5/seinfeld-chronicles#scripts.csv) of scripts from 9 seasons.
The Neural Network we'll build will generate a new ,\"fake\" TV script, based on patterns it recognizes in this training data.

### Preprocessing
The first thing to do to any dataset is pre-processing. We implement the following pre-processing functions below:

- To create a word embedding, we first needed to transform the words to ids. In ```create_lookup_tables(text)```, we created two dictionaries:
  - Dictionary to go from the words to an id, we'll call vocab_to_int.
  - Dictionary to go from the id to word, we'll call int_to_vocab.
Return these dictionaries in the following tuple (vocab_to_int, int_to_vocab)

- In  ```token_lookup()```, we splitted the script into a word array using spaces as delimiters. However, punctuations like periods and exclamation marks 
can create multiple ids for the same word. For example, "bye" and "bye!" would generate two different word ids. We implemented the function token_lookup to return a dict that will be used to tokenize symbols like "!" into
"||Exclamation_Mark||". We created a dictionary for the following symbols where the symbol is the key and value is the token:
  - Period ( . )
  - Comma ( , )
  - Quotation Mark ( " )
  - Semicolon ( ; )
  - Exclamation mark ( ! )
  - Question mark ( ? )
  - Left Parentheses ( ( )
  - Right Parentheses ( ) )
  - Dash ( - )
  - Return ( \n )
  
This dictionary is used to tokenize the symbols and add the delimiter (space) around it. This separates each symbols as its 
own word, making it easier for the neural network to predict the next word. We made sure that we don't use a value that could be 
confused as a word; for example, instead of using the value "dash", we used something like "||dash||".

### Model Training
- Batching: We implemented the batch_data function to batch words data into chunks of size batch_size using the TensorDataset 
and DataLoader classes.

- Model: We used the RNN class we implemented to apply forward and back propagation. This function is called, iteratively, 
in the training loop as : ```loss = forward_back_prop(decoder, decoder_optimizer, criterion, inp, target)```
And it returns the average loss over a batch and the hidden state returned by a call to ```RNN(inp, hidden)```. 
We get this loss by computing it, as usual, and calling ```loss.item()```.

- Model Hyperparameters
  - Sequence Length : Average number of words per sentence was around 5.5. I started with a sequence length of 10 but later changed it to 8 in order to get nearer to average number of words. Experiments showed that loss was decreasing a bit quickly.
  - Epochs : I decided to first try 10 epochs. Loss continued to decrese. I did not train the model further as the requirements are met and time and compute resources are limited.
  - Batch size: Randomly started with 256.
  - Learning Rate: I used the Adam optimizer and tried 0.1 learning rate. Loss was increasing so I decided to go with 0.001 which resulted in effiient learning.
  - n_layers: As told in the lesson, 2 layers are better than 1 and 3 layers give mixed results. Currently, with 2 layers model is not very complex.

Other hyperparameters were chosen randomly. Thorough testing would be helpful for better performance, provided the needed resources.

Please, have a look [here](https://github.com/UsmanIjaz/DL_TV_Script_Generation/blob/master/dlnd_tv_script_generation.ipynb) for implementation details.


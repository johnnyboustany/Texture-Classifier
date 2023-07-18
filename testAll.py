import numpy as np
import tensorflow as tf
from preprocess import get_data
from tensorflow.keras import Model
import os
import random
import math
# ensures that we run only on cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class Model(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()

        # initialize embedding_size, batch_size, and any other hyperparameters
        self.vocab_size = vocab_size
        self.window_size = 20
        self.embedding_size = 30  # TODO
        self.batch_size = 200  # TODO
        self.E = tf.Variable(tf.random.normal((vocab_size, self.embedding_size), stddev=.1,dtype=np.float32))  ## TODO
        self.LSTM=tf.keras.layers.LSTM(100,return_sequences=True,return_state=True)
        self.dense=tf.keras.layers.Dense(self.vocab_size,activation='softmax')
  
    def call(self, inputs, initial_state):
        """
        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state
        """
        out,h_t,s_t=self.LSTM(tf.nn.embedding_lookup(self.E,inputs),initial_state=initial_state)
        probabilities=self.dense(out)
        return probabilities,(h_t,s_t)

    def loss(self, probabilities, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probabilities: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the average loss of the model as a tensor of size 1
        """

        # TODO: Fill in
        # We recommend using tf.keras.losses.sparse_categorical_crossentropy
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities, axis=-1))
        return None


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples 

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,) reshape to be batch_size, window_size
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    c = list(zip(train_inputs, train_labels))
    random.shuffle(c)
    trainInputs, trainLabels = zip(*c)
    leng=int(len(trainInputs)/model.window_size)
    trainInputs=trainInputs[:-(len(trainInputs)%model.window_size)]
    trainLabels=trainLabels[:-(len(trainLabels)% model.window_size)]
    trainInputs=np.reshape(trainInputs,(-1,model.window_size))
    trainLabels=np.reshape(trainLabels,(-1,model.window_size))
    for i in range(int(len(train_inputs)/model.batch_size)):
        with tf.GradientTape() as tape:
            trainOutput1,tup = (model.call(trainInputs[i*model.batch_size:(i+1)*model.batch_size],initial_state = None))
            #tf.argmax
            # Forward pass
            loss = model.loss(trainOutput1,trainLabels[i*model.batch_size:(i+1)*model.batch_size])
        gradients = tape.gradient(loss, model.trainable_variables)
        tf.keras.optimizers.Adam(learning_rate=.01).apply_gradients(zip(gradients, model.trainable_variables))
    # TODO: Fill in
    pass


def test(model, test_inputs, test_labels):
    """
    Runs through one epoch - all testing examples 

    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set
    """

    trainInputs=test_inputs[:-(len(test_inputs)%model.window_size)]
    trainLabels=test_labels[:-(len(test_labels)% model.window_size)]
    trainInputs=np.reshape(trainInputs,(-1,model.window_size))
    trainLabels=np.reshape(trainLabels,(-1,model.window_size))
    acc=0
    a=int(len(trainInputs) / model.batch_size)
    for i in range(int(len(trainInputs)/model.batch_size)):
        trainOutput1,tup=model.call(trainInputs[i*model.batch_size:(i+1)*model.batch_size],initial_state = None)
        acc+=model.loss(trainOutput1, trainLabels[i*model.batch_size:(i+1)*model.batch_size])/a
    return np.exp(acc)
    pass


def generate_sentence(word1, length, vocab, model, sample_n=10):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx: word for word, idx in vocab.items()}
    previous_state = None

    first_string = word1
    first_word_index = vocab[word1]
    next_input = [[first_word_index]]
    text = [first_string]

    for i in range(length):
        logits, previous_state = model.call(next_input, previous_state)
        logits = np.array(logits[0, 0, :])
        top_n = np.argsort(logits)[-sample_n:]
        n_logits = np.exp(logits[top_n]) / np.exp(logits[top_n]).sum()
        out_index = np.random.choice(top_n, p=n_logits)

        text.append(reverse_vocab[out_index])
        next_input = [[out_index]]

    print(" ".join(text))


def main():
    # Pre-process and vectorize the data
    x,y,z = get_data("train.txt","test.txt")
    trainData=x[0:len(x)-1]
    trainLab=x[1:len(x)]
    testData=y[0:len(y)-1]
    testLab=y[1:len(y)]
    model=Model(len(z))
    train(model, trainData, trainLab)
    print(test(model,testData,testLab))
    pass


if __name__ == "__main__":
    main()

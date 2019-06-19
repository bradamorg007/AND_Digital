from multi_channel_cnn.preprocessing import DataOrganiser
import numpy as np
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import Embedding

from keras.optimizers import rmsprop
import math

from keras.layers.convolutional import Conv1D
from keras.layers import GlobalMaxPool1D
from keras.layers.merge import concatenate
import os

import matplotlib.pyplot as plt

# Build the embedding matrix: Use googles pre-trained vector representations
def define_embedding_matrix(word_index):

    # extract words and vectors from glove.txt
    embedding_dims = 100
    dir = "../glove.6B/"
    fileName = "glove.6B.100d.txt"

    embedding_index = {}
    file = open(os.path.join(dir, fileName), encoding="utf8")

    for txt_line in file:
        vals = txt_line.split()
        word_tag = vals[0]
        vector_rep = np.asarray(vals[1:], dtype='float32')
        embedding_index[word_tag] = vector_rep
    file.close()

    print('Program has detected %s word vectors from the %s embedding file' % (len(embedding_index), fileName))

    # define embedding matrix and apply googles vectors to my datasets vocab
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dims))
    count = 0
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            count = count + 1

    print('Number of detected matches between embedding index and data set vocab = %s' % count)
    return embedding_matrix, embedding_dims


# Associate the unique class labels with unique one-hot encodings: e.g debt collection = [0,0,0,1,0,0]
def bind_label_encodings(encodings, input):
    return array([encodings.encoded_label_map.get(item) for item in input])


# Updates internal vocabulary based on a list of texts. This method creates the vocabulary index based on word frequency.
def tokenize(word_length, num_words, input):
    # word length is the maximum number of words each review will contain
    # num_words the number of most commonly occuring words that will be considered e.g 10k most frequent words fr vocab
    # adds out text data to the keras token vocab and converts the text to ints
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(input)

    # turn words into integers
    integer_mappings = tokenizer.texts_to_sequences(input)

    # integer token sequences to the max length of words for each customer complaint document
    padded_sequences = pad_sequences(sequences=integer_mappings, maxlen=word_length,
                                     padding='post')

    print('Max word length for customer complaint documents = %s' % word_length)
    print('Vocabulary size in dataset = %s' % (len(tokenizer.word_index) + 1))
    print('Number of most frequent words to be considered from vocab = %s' % tokenizer.num_words)
    return tokenizer, padded_sequences, len(tokenizer.word_index) + 1

# Randomise data ordering to stop overfitting in the model
def shuffle_data(x, y):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)

    return x[indices], y[indices]

# Plot the training and validation accuracy and loss
def plot_results(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure()
    plt.plot(epochs, acc, 'g', label='training acc')
    plt.plot(epochs, val_acc, 'r', label='validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'g', label='training loss')
    plt.plot(epochs, val_loss, 'r', label='validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# Define the model: Based on research paper Yoon Kim (2014) https://arxiv.org/abs/1408.5882
def multichannel_cnn1_model(x, y, word_len, vocab_size, embedding_matrix, embedding_dims):
    # MODEL
    #
    # One embedding will be used to feed to each N-gram channel

    loadModel = False

    if loadModel == False:
        inputs = Input(shape=(word_len, ))

        EL1 = create_cnn_embedding_layer(inputs, vocab_size, embedding_dims, word_len, [], True, name='embed1')
        EL2 = create_cnn_embedding_layer(inputs, vocab_size, embedding_dims, word_len, [], True, name='embed2')
        EL3 = create_cnn_embedding_layer(inputs, vocab_size, embedding_dims, word_len, [], True, name='embed3')
        EL4 = create_cnn_embedding_layer(inputs, vocab_size, embedding_dims, word_len, [], True, name='embed4')
        EL5 = create_cnn_embedding_layer(inputs, vocab_size, embedding_dims, word_len, [], True, name='embed5')
        EL6 = create_cnn_embedding_layer(inputs, vocab_size, embedding_dims, word_len, [], True, name='embed6')


        # channel 1
        chan1 = create_cnn_channel(input=EL1, filters=50, kernel_size=3)

        # channel 2
        chan2 = create_cnn_channel(input=EL2, filters=50, kernel_size=8)

        # # channel 3
        chan3 = create_cnn_channel(input=EL3, filters=50, kernel_size=13)

        # # channel 4
        chan4 = create_cnn_channel(input=EL4, filters=50, kernel_size=18)

        # # channel 5
        chan5 = create_cnn_channel(input=EL5, filters=32, kernel_size=25)
        #
        # # channel 6
        chan6 = create_cnn_channel(input=EL6, filters=32, kernel_size=32)


        # Merge / concat channel outputs fbg
        merge = concatenate(inputs=[chan1, chan2, chan3, chan4, chan5, chan6])
        dropout = Dropout(rate=0.8)(merge)

        # Feed into dense layers for prediction
        dense1 = Dense(75, activation='relu')(merge)
        dropout2 = Dropout(rate=0.2)(dense1)
        outputs = Dense(len(y[0]), activation='softmax')(dropout2)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(loss='categorical_crossentropy', optimizer=rmsprop(), metrics=['accuracy'])

    else:

        model = load_model('../multi_channel_cnn/models/model_complex_best.h5')

    print(model.summary())


    history = model.fit(x=x, y=y, epochs=10, batch_size=200, validation_split=0.1, verbose=2)
                       # callbacks=[callbacks.TensorBoard(log_dir='../multi_channel_cnn/logs/', histogram_freq=1,
                                                       #  embeddings_freq=1, embeddings_data=x[:100])])
                        #callbacks=[EarlyStopping(monitor='val_loss', patience=4, min_delta=0.0001)])
    model.save('model_complex_Best_V2.h5')
    plot_results(history)


def create_cnn_channel(input, filters, kernel_size):


    conv3 = Conv1D(filters=filters, kernel_size=kernel_size, activation='relu')(input)
    dropout3 = Dropout(rate=0.25)(conv3)
    globalMaxPool = GlobalMaxPool1D()(dropout3)

    return globalMaxPool



def create_cnn_embedding_layer(inputs, vocab_size, embedding_dims, max_word_len, weight_matrix, trainable, name):

    if len(weight_matrix) == 0:
        embedding_layer = Embedding(vocab_size, embedding_dims,
                                    input_length=max_word_len, name=name)(inputs)

    else:
        embedding_layer = Embedding(vocab_size, embedding_dims,
                                input_length=max_word_len, weights=[weight_matrix], trainable=trainable,
                                    name=name)(inputs)

    return embedding_layer

# calculate the maximum document length
def max_length(lines):

    max = 0
    sum = 0
    ls = []
    for s in lines:
        l = len(s.split())

        if l > max:
            max = l

        sum = sum + l
        ls.append(l)

    plt.figure()
    plt.plot(ls)
    plt.title('Word lengths of documents')

    return max, math.floor(sum/len(lines))


if __name__ == "__main__":
    data = DataOrganiser()
    data.load_data('cc_BIG1.2_dataset.do.pkl')

    (x_train, y_train) = data.asarray(data.train)

    print('Training Dataset sample size = %s' % len(x_train))
    word_len, avg_word_len = max_length(x_train)

    # Tokenize data, add to vocab and pad to max word length
    tokenizer, x_train, vocab_size = tokenize(word_length=avg_word_len, num_words=20000, input=x_train)

    embedding_matrix, embedding_dims = define_embedding_matrix(word_index=tokenizer.word_index)

    # assign one_hot encodings to y_train and y_test labels
    y_train = bind_label_encodings(data, y_train)

    # shuffle data
    x, y = shuffle_data(x_train, y_train)

    multichannel_cnn1_model(x=x, y=y, word_len=avg_word_len, vocab_size=vocab_size,
                            embedding_matrix=embedding_matrix, embedding_dims=embedding_dims)

# AND_Digital
Multi Channel 1D Convolutional Neural Network for Text Classification

# Use Case

Text classification can be used to automate customer support 

# Define Model
A standard model for document classification is to use an Embedding layer as input, followed by a one-dimensional convolutional neural network, pooling layer, and then a prediction output layer.

The kernel size in the convolutional layer defines the number of words to consider as the convolution is passed across the input text document, providing a grouping parameter.

A multi-channel convolutional neural network for document classification involves using multiple versions of the standard model with different sized kernels. This allows the document to be processed at different resolutions or different n-grams (groups of words) at a time, whilst the model learns how to best integrate these interpretations.

This approach was first described by Yoon Kim in his 2014 paper titled “Convolutional Neural Networks for Sentence Classification.”

In the paper, Kim experimented with static and dynamic (updated) embedding layers, we can simplify the approach and instead focus only on the use of different kernel sizes.

This approach is best understood with a diagram taken from Kim’s paper:

Depiction of the multiple-channel convolutional neural network for text
Depiction of the multiple-channel convolutional neural network for text.
Taken from “Convolutional Neural Networks for Sentence Classification.”

In Keras, a multiple-input model can be defined using the functional API.

We will define a model with three input channels for processing 4-grams, 6-grams, and 8-grams of movie review text.

# Each channel is comprised of the following elements:

Input layer that defines the length of input sequences.
Embedding layer set to the size of the vocabulary and 100-dimensional real-valued representations.
One-dimensional convolutional layer with 32 filters and a kernel size set to the number of words to read at once.
Max Pooling layer to consolidate the output from the convolutional layer.
Flatten layer to reduce the three-dimensional output to two dimensional for concatenation.
The output from the three channels are concatenated into a single vector and process by a Dense layer and an output layer.

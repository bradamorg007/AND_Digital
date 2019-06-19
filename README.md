# AND_Digital
Multi Channel 1D Convolutional Neural Network for Text Classification

# Use Case

Text classification can be used to automate customer support in many areas, such as call routing, feedback and complaint catogorisation and sentiment analysis of customer reviews. The code outlined here demonstrates how machine learning models can be affectively applied to these areas via multichanneled 1D convolutional neural network. The data set used for this demonstaration is comprised of 126352 customer complaints to an annonymous banking company, where the complaints have been labled to their associated catogories, such as mortgages, debt collection etc. We can leverage machine learning to predict which topic or department the complaint should be routed towards. The same model and approuch can be applied to golden shoe to aid with customer support workers with feedback and complaint/request analysis.
Please note that the data set and embeddings are too large to host GITHUB, so I have only included the python files for the model and pre-processing classes.
Please find the data set at this URL https://www.kaggle.com/kharaldsson/consumter-complaints
Please find embeddings at https://nlp.stanford.edu/projects/glove

The code will not run without the above. Thanks for reading

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
One-dimensional convolutional layer with 50 filters and a kernel size set to the number of words to read at once known as N-grams.
Max Pooling layer to consolidate the output from the convolutional layer.
Flatten layer to reduce the three-dimensional output to two dimensional for concatenation.
The output from the three channels are concatenated into a single vector and process by a Dense layer and an output layer.

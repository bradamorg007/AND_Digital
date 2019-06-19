import pandas as pd
import numpy as np
import math
import os
from string import punctuation
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import time
import re
import sys
from pickle import dump
from pickle import load

# This class provides static methods for Cleaning text based data for NLP
class DataCleaner():

    def __init__(self):
        pass

    # Method Cleans High level aspects of the data set such as removing blank or NaN entry values
    # Method extracts relevant columns from data set table. Selects valid/usable sample points
    @staticmethod
    def data_clean(path, removeCols, cut_percentage, randomise):
        print('============== Cleaning Text Data ======================')
        dataframe = pd.read_csv(path, header=0)

        # remove unwanted columns

        cols = []
        for x in range(len(dataframe.columns)):
            match = False

            for y in removeCols:
                if x == y:
                    match = True
                    break

            if match == False:
                cols.append(x)

        dataframe.drop(dataframe.columns[cols], axis=1, inplace=True)
        dataframe.columns = ['route', 'complaint']

        # convert to numpy array
        data_arr = np.asarray(dataframe.values)

        # remove any nan values
        data = []
        for row in data_arr:

            match = False
            for col in row:

                if isinstance(col, float):
                    if math.isnan(col):
                        match = True
                        break

            if match == False:
                data.append(row)

        data = np.asarray(data)
        print('%s valid data samples detected' % len(data))

        cutoffPoint = math.floor(len(data) * cut_percentage)

        if randomise == 'random' and cut_percentage < 1:
            inds = np.random.randint(0, len(data), size=cutoffPoint)
            data = data[inds]

        elif randomise == 'from_start':
            data = data[:cutoffPoint]
        else:
            raise ValueError('Please selected valid last argument either randomise or from_start')

        print(' cutoff percentage = %s samples = %s  selection = %s' % (cut_percentage, len(data), randomise))

        # global treatment on data and labels
        tokens = []
        count = 0

        for text_sample in data:
            begin = time.time()
            tokens.append(DataCleaner.token_clean(token_pair=text_sample))
            count = count + 1
            DataCleaner.progressBar(count, len(data), 20)

        print('\n============== DONE! ======================')
        return tokens

    # displays processing progress
    @staticmethod
    def progressBar(value, endvalue, bar_length=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

    #Cleans Individual text data of sample points
    @staticmethod
    def token_clean(token_pair):
        if len(token_pair) == 2:
            # assume token pair is ordered {label, text data}
            token_data = {}
            for i, token in enumerate(token_pair):
                if (isinstance(token, str)):
                    # This maps all punctutation values to empty strs
                    punc_map = str.maketrans('', '', punctuation)
                    token = str.lower(token)

                    # remove censorship tags like XXXX or XX
                    token = re.sub(r'([x])\1+', '', token)

                    # remove whitespace
                    # token = token.split()
                    token = nltk.word_tokenize(token)

                    # remove punctuation
                    w = []
                    for word in token:
                        w.append(word.translate(punc_map))
                    token = w

                    # remove all non-alphabetic words
                    w = []
                    for word in token:
                        if word.isalpha():
                            w.append(word)
                    token = w

                    # remove all stop words
                    w = []
                    stop_words = set(stopwords.words('english'))
                    for word in token:
                        if not word in stop_words:
                            w.append(word)
                    token = w

                    # Remove one char len words
                    w = []
                    for word in token:
                        if len(word) > 1:
                            w.append(word)
                    token = w

                    # lemmatize words e.g wolves & wolf = wolf - same word just plural I dont think it will be needed
                    # init lementizer
                    lemmatizer = nltk.stem.WordNetLemmatizer()
                    w = []
                    for word in token:
                        w.append(lemmatizer.lemmatize(word))

                    token = w

                    # do final sweep check using word net, this will remove any words that is not in the wordnet database. get rid
                    # of any other words we forgot about
                    w = []
                    for word in token:
                        if wordnet.synsets(word):
                            # this word exists in database
                            w.append(word)

                    token = w
                    word_len = len(token)

                    # Join them back to strings so toaavoid any issues with keras tokensier - kind weird but just to be on the safe side
                    token = ' '.join(token)

                    if i == 0:
                        token_data['label'] = token
                        token_data['label_word_length'] = word_len
                    elif i == 1:
                        token_data['text'] = token
                        token_data['text_word_length'] = word_len

                else:
                    raise ValueError('The Input to clean token must be a string object')

            return token_data

        else:
            raise ValueError('input must be a token pair comprised of a label and text training data')

    @staticmethod
    def save_data(data, filename):
        if filename[-6:] != '.dc.pkl':
            filename = filename + '.dc.pkl'

        path = '../multi_channel_cnn/clean_data/'

        if os.path.exists(path) == False:
            os.mkdir(path)
            dump(data, open(os.path.join(path, filename), 'wb'))

        else:
            dump(data, open(os.path.join(path, filename), 'wb'))

        print('DataCleaner Saved Itself: %s ' % filename)

    @staticmethod
    def load_data(filename):
        if filename[-6:] != '.dc.pkl':
            raise ValueError("Incorrect file extention please load pickle file")

        path = os.path.join(
            '../multi_channel_cnn/clean_data/',
            filename)

        if os.path.exists(path):
            data = load(open(path, 'rb'))
            print('Load Complete: %s ' % filename)
            return data

        else:
            raise ValueError("The filedoes not exist or is not located in the clean_data directory")


class DataOrganiser():

    def __init__(self):
        self.data = None
        self.label_dict = None
        self.label_map = None
        self.encoded_label_map = None
        self.train = None
        self.validation = None
        self.test = None
        self.vocab_size = None

    def load_data(self, filename):

        path = os.path.join(
            '../multi_channel_cnn/clean_data/',
            filename)

        if os.path.exists(path):

            if filename[-6:] == 'do.pkl':
                data = load(open(path, 'rb'))
                print('Load Complete: %s ' % filename)
                self.data = data.data
                self.label_dict = data.label_dict
                self.label_map = data.label_map
                self.encoded_label_map = data.encoded_label_map
                self.train = data.train
                self.test = data.test
                self.validation = data.validation

            elif filename[-6:] == 'dc.pkl':
                data = load(open(path, 'rb'))
                print('Load Complete: %s ' % filename)
                self.data = data

            else:
                raise ValueError("Incorrect file extention please load a dc.pkl or do.pkl file")


        else:
            raise ValueError("The filedoes not exist or is not located in the clean_data directory")

    def save_data(self, filename):
        if filename[-6:] != '.do.pkl':
            filename = filename + '.do.pkl'
        path = '../multi_channel_cnn/clean_data/'

        if os.path.exists(path) == False:
            os.mkdir(path)
            dump(self, open(os.path.join(path, filename), 'wb'))
            print('DataOrganiser Saved Itself: %s ' % filename)

        else:
            dump(self, open(os.path.join(path, filename), 'wb'))
            print('DataOrganiser Saved Itself: %s ' % filename)

    # Min and maximum occurring label. e.g label A has 90 sample points and label B has 30
    def max_label_freq(self):

        max = 0

        for label in self.label_dict.items():

            if label[1]['freq'] > max:
                max = label[1]['freq']

        return max

    def min_label_freq(self):

        min = 0

        for label in self.label_dict.items():

            if min == 0:
                min = label[1]['freq']
            elif label[1]['freq'] < min:
                min = label[1]['freq']

        return min

    # Min and max word length for customer complaint document
    def max_word_length(self):

        max = 0

        for label in self.data:

            if label['text_word_length'] > max:
                max = label['text_word_length']

        return max

    def min_word_length(self):

        min = 0

        for label in self.data:

            if min == 0:
                min = label['text_word_length']

            elif label['text_word_length'] < min:
                min = label['text_word_length']

        return min

    # Creates a dictionary storing each class label as a unique key. The values of these keys are the associated
    # text documents that represents the customers complaints
    def organise(self, simplify):
        # need to determine unique occurrences of label words and tally the occurrences and the indices where they occured
        # to do this I will use a dict within a dict for fast lookup and insert times

        self.data = np.asarray(self.data)

        table = DataOrganiser.count_unquie(self.data)

        key_map = {}
        new_table = {}
        if simplify:
            # reduces multi_word labels into single word acronyms store full word in dict
            for (old_key, old_elem) in table.items():

                # create new key
                key_str = old_key.split()
                new_key = []
                for word in key_str:
                    new_key.append(word[0])

                new_key = ''.join(new_key)
                # create reverse mapping
                key_map[new_key] = old_key
                # update table keys
                new_table[new_key] = old_elem
                # update data array aswell

                for i in old_elem['indices']:
                    self.data[i]['label'] = new_key
                    self.data[i]['label_word_length'] = 1

            self.label_dict = new_table
            self.label_map = key_map


        else:
            self.label_dict = table


    # To reduce model overfitting reduce the data to equal split of class labels. Reduce to the min threshold of samples
    # required. e.g Class A, samples = 1000  classB, samples = 500. Reduce will crop Class A & B to 500 samples each
    # updates all other dictionaries that map relevant sample locations and
    def reduce(self, min_samples):

        # Crop samples to a min number of samples, all sample pools will be cut to this size
        # split remaining data into test val and training data pools, make sure percentages add to 1
        # then make test, val and train all into arrays and not dicts

        if min_samples > self.max_label_freq():
            raise ValueError('ERROR: Reduce Function DataOrganiser, the min_samples can not be larger than the maximum label frequencyu')

        min_samples = math.floor(min_samples)

        # find min at same time
        min = 0
        keys = self.label_dict.keys()
        reduced_label_dict = {}

        reduced_label_map = {}
        for key in keys:

            current_label_dict = self.label_dict[key]

            if current_label_dict['freq'] >= min_samples:

                reduced_label_dict[key] = current_label_dict

                if self.label_map != None:
                    reduced_label_map[key] = self.label_map[key]

                if min == 0:
                    min = current_label_dict['freq']

                elif current_label_dict['freq'] < min:
                    min = current_label_dict['freq']

        # Now crop label dict sample indices down to the min freq value
        keys = reduced_label_dict.keys()
        for key in keys:

            current_label_dict = reduced_label_dict[key]

            if current_label_dict['freq'] > min:
                # by how much
                diff = current_label_dict['freq'] - min
                # randomly choose  number of items to keep whilst removing diff amount of items from  list
                indices = np.random.randint(0, current_label_dict['freq'], current_label_dict['freq'] - diff)

                current_label_dict['indices'] = [current_label_dict['indices'][i] for i in indices]
                current_label_dict['freq'] = len(current_label_dict['indices'])

        # Now make a new data set with only the reduced lists
        pull_indices = []
        tally = 0
        for key in keys:
            dict = reduced_label_dict[key]
            pull_indices = pull_indices + dict['indices']

            dict['indices'] = np.arange(tally, tally + len(dict['indices'])).tolist()
            tally = tally + len(dict['indices'])

        self.data = self.data[pull_indices]
        self.label_dict = reduced_label_dict

        if self.label_map != None:
            self.label_map = reduced_label_map


    # Allows for one to manually split dataset into training, test and validation proportions either by simply via
    # random selection or by selecting equal class label sample amounts relative to train, test,validation split percentages
    def train_val_split(self, train_split, val_split, test_split, pick_type='random'):

        if train_split + val_split + test_split == 1:

            # randomise order of the data and split into 3 chunks
            indices = np.arange(self.data.size)

            t = math.floor(len(indices) * train_split)
            v = math.floor(len(indices) * val_split)
            te = math.floor(len(indices) * test_split)

            if pick_type == 'random':
                np.random.shuffle(indices)

                train_data = indices[0:t]
                val_data = indices[t: t + v]
                test_data = indices[t + v:t + v + te]

                self.train = self.data[train_data]
                self.validation = self.data[val_data]
                self.test = self.data[test_data]

            elif pick_type == 'equal':
                t = math.floor(t / len(self.label_dict))
                v = math.floor(v / len(self.label_dict))
                te = math.floor(te / len(self.label_dict))

                t_indices = []
                v_indices = []
                te_indices = []

                keys = self.label_dict.keys()
                for key in keys:

                    dict_indices = np.asarray(self.label_dict.get(key).get('indices'))

                    t_indices = t_indices + dict_indices[0:t].tolist()
                    v_indices = v_indices + dict_indices[t:t + v].tolist()
                    te_indices = te_indices + dict_indices[t + v:t + v + te].tolist()

                if self.valid(self.data[t_indices]) and self.valid(self.data[v_indices]) and self.valid(self.data[te_indices]):
                    self.train = self.data[t_indices]
                    self.validation = self.data[v_indices]
                    self.test = self.data[te_indices]
                else:
                    raise ValueError('Error: Function failure: Detected non-equal train test val split quantities')


        else:
            raise ValueError('Error: split must be a 3D vector with percentage values that sum to 1')


    # Helper method that checks that there is indeed and equal distribution of class labels in the data set
    def valid(self, input):

        table = DataOrganiser.count_unquie(input)

        keys = table.keys()

        mark = 0
        for i, key in enumerate(keys):

            freq = table.get(key).get('freq')

            if i == 0:
                mark = freq
            elif freq != mark:
                return False

        return True


    # transforms word/text class labels into one hot encoding vectors for softmax prediction layer in model
    def encode_labels(self):

        labels = list(self.label_dict.keys())
        dummies = pd.get_dummies(labels).values

        self.encoded_label_map = dict(zip(labels, dummies))

    def print_labels(self):

        keys = self.label_dict.keys()

        for key in keys:
            print('key: %s  sample size: %s' % (key, (self.label_dict.get(key)).get('freq')))


    # Uses hash mapping via dictonary lookups to quickly tally the samples to their relevent labels
    @staticmethod
    def count_unquie(input):
        table = {}

        for i, rows in enumerate(input):

            # try to retrieve a label key, if it exists append the freq count and add i to indices

            lookup = table.get(rows['label'])

            if lookup is not None:

                lookup['freq'] += 1
                lookup['indices'].append(i)

            else:
                # if it doesnt exist in the dict then make a new entry
                table[rows['label']] = {'freq': 1, 'indices': [i]}

        return table

    # converts data to numpy arrays as they are much easier to handle and index with
    @staticmethod
    def asarray(input):

        text = []
        labels = []

        for dict in input:
            text.append(dict.get('text'))
            labels.append(dict.get('label'))

        return np.array(text), np.array(labels)


# Main statement that runs the cleaning and organising process. Please not that data cleaning can on take sometime
# depending on your computer specs
if __name__ == "__main__":

    path ='../multi_channel_cnn/unclean_data/Consumer_Complaints_BIG1.csv'
    remove_cols=[1, 5]
    cut_percentage= 1
    randomise='from_start'
    save_name='cc_BIG1_dataset'
    simplify=True

    data = DataCleaner.data_clean(path=path, removeCols=remove_cols, cut_percentage=cut_percentage, randomise=randomise)
    DataCleaner.save_data(data=data, filename=save_name)

    data_orger = DataOrganiser()
    data_orger.load_data(filename='cc_BIG1_dataset.dc.pkl')
    data_orger.organise(simplify=False)
    print()
    print('label with highest number of associated text samples = %s' % data_orger.max_label_freq())
    print('label with least number of associated text samples = %s' % data_orger.min_label_freq())
    print('maximum word length for all text samples in datasdet = %s' % data_orger.max_word_length())
    print('minimum word length for all text samples in datasdet = %s' % data_orger.min_word_length())
    print()
    data_orger.reduce(min_samples=25000)
    data_orger.encode_labels()
    data_orger.train_val_split(train_split=1, val_split=0.0, test_split=0.0, pick_type='equal')
    data_orger.save_data(filename='cc_BIG1.2_dataset')
    print('NUMBER OF LABELS SELECTED = %s' % len(data_orger.label_dict))
    print('SAMPLE SIZE = %s' % len(data_orger.data))
    data_orger.print_labels()

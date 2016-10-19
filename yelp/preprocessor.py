#preprocessor
#from spacy.en import English
from os import listdir
from os.path import isfile, join
import numpy as np
import json

# nlp = English() # for use with spaCy

class DatasetJson(object):
    def __init__(self, filename, text_field, response_field, limit = 25, maxlen = 10, max_features = 5000):
        #spacy = __import__('spacy')
        #vocab = VocabularyChar()

        self.filename = filename
        self.text_field = text_field
        self.response_field = response_field
        self.maxlen = maxlen

        # read json filename ()
        self.read_json(limit)
        # split response from data
        self.extract_data()

    def read_json(self,limit=25):
        i = 0;
        data = []
        with open(self.filename, "rb") as infile:
            for line in infile:
                #print(line)
                i = i + 1
                data.append(json.loads(line))
                if(i == limit): break

        self.data = data

    def extract_data(self):
        text = []
        response = []
        for idx,line in enumerate(self.data):
            text.append(self.data[idx][self.text_field])
            response.append(self.data[idx][self.response_field])
        self.text = text
        self.response = response
        self.data = "" # save space

    def create_vocab(self,sentences):
        words = []
        for row in sentences:
            for i in row:
                #exclude = set(i.punctuation)
                #s = ''.join(ch for ch in s if ch not in exclude)
                words.append(i.lower())
        return {word: idx for idx, word in enumerate(set(words))}

    def hot_vector_sequences(self,sequences,vocab,maxlen=40):
        vec_len = len(vocab)
        new_sequences = []
        for sequence in sequences: # sequence here is a list of words
            orig_len = np.shape(sequence)
            orig_len = orig_len[0]
            # first create a numpy array of arrays
            new_seq = []
            for word in sequence:
                word_vec = np.zeros(vec_len)
                word_vec[vocab[word.lower()]] = 1 # hot vector
                new_seq.append(word_vec)

            sequence = new_seq
            # now replace the sequence with new word vec matrix
            # out of laziness :(

            if orig_len < maxlen:
                new = np.zeros((maxlen,vec_len))
                new[maxlen-orig_len:] = sequence
            else:
                new = sequence[orig_len-maxlen:]
            new_sequences.append(new)
        return np.array(new_sequences)

    def index_sequences(self,sequences,vocab,maxlen=40,max_features=5000):
        vec_len = len(vocab)
        new_sequences = []

        for sequence in sequences: # sequence here is a list of words
            orig_len = np.shape(sequence)
            orig_len = orig_len[0]
            # first create a numpy array of arrays
            new_seq = []
            for word in sequence:
                vocab_position = vocab[word.lower()]
                if(vocab_position < max_features):
                    new_seq.append(vocab_position) # create vector of vocabulary indices
                else:
                    new_seq.append(0) # add zero feature is out of position

            sequence = new_seq
            # now replace the sequence with new index vector
            # out of laziness :(

            # padding
            if orig_len < maxlen:
                new = np.zeros(maxlen)
                new[maxlen-orig_len:] = sequence
            else:
                new = sequence[orig_len-maxlen:]
            new_sequences.append(new)
        return np.array(new_sequences)

    def hot_TDM(self,sequences,vocab,maxlen=40):
        # this tdm is not weighted in any way
        vec_len = len(vocab)
        tdm = []

        for sequence in sequences: # sequence here is a list of words
            orig_len = np.shape(sequence)
            orig_len = orig_len[0]

            # initialise doc and word vectors
            doc_vec = np.zeros(vec_len)

            # the doc vec is just the sum of the word vectors
            for word in sequence:
                word_vec = np.zeros(vec_len)
                word_vec[vocab[word.lower()]] = 1 # hot vector
                doc_vec = doc_vec + word_vec # there are quicker way to do this, but this is easier to read?
                # new_seq.append(word_vec)

            # put the doc vectors together
            tdm.append(doc_vec)

        # one could add IDF weighting here
        # this is actaully a document term matrix as its documents X terms but this is fine it will fit straight into a model
        return np.array(tdm)

class DocuemntTermMatrix(DatasetJson):
    def __init__(self, filename, text_field, response_field, limit = 25, maxlen = 10, max_features = 5000):
        # quick setup
        self.filename = filename
        self.text_field = text_field
        self.response_field = response_field
        self.maxlen = maxlen

        # read json filename ()
        #  sets up self.data
        self.read_json(limit)
        # split response from data
        #  sets up self.text and self.respone (labels)
        self.extract_data()

        # extract labels and text

        # set up containers
        X_docs = []
        Y_docs = []

        # extract words for docs and sentences
        for idx, response in enumerate(self.response):

            doc = self.text[idx]

            # split into words and sentences
            x_words  = unicode(doc).split(" ")

            # each document is an observation (this is a traditional TDM format)
            if len(x_words) > 0:

                X_docs.append(np.array(x_words))
                Y_docs.append(response)

        # create response
        self.Y_docs      = Y_docs

        # create an index for the response (this will make it easier to create a response matrix later)
        self.docs_label_index         = {label :idx for idx,label in enumerate(set(Y_docs),start=0) }

        # create vocab for both points of view
        self.docs_vocab      = self.create_vocab(X_docs)

        # create sequence matrix for data
        self.X_docs      = self.hot_TDM(X_docs,self.docs_vocab,maxlen)

class Index_Sequence(DatasetJson):
    def __init__(self, filename, text_field, response_field, limit = 25, maxlen = 10, max_features = 5000):
        # quick setup
        self.filename = filename
        self.text_field = text_field
        self.response_field = response_field
        self.maxlen = maxlen

        # read json filename ()
        self.read_json(limit)
        # split response from data
        self.extract_data()

        # extract labels and text

        # set up containers
        X_doc_seq = []
        Y_doc_seq = []

        # extract words for docs and sentences
        for idx, response in enumerate(self.response):

            doc = self.text[idx]

            # split into words and sentences
            x_words  = unicode(doc).split(" ")

            # each document is an observation (this is a traditional TDM format)
            if len(x_words) > 0:

                X_doc_seq.append(np.array(x_words))
                Y_doc_seq.append(response)

        # create response
        self.Y_doc_seq        = Y_doc_seq

        # create an index for the response (this will make it easier to create a response matrix later)
        self.docs_label_index = {label :idx for idx,label in enumerate(set(Y_doc_seq),start=0) }

        # create vocab for both points of view
        self.docs_vocab = self.create_vocab(X_doc_seq)

        # create sequence matrix for data
        self.X_doc_seq  = self.index_sequences(X_doc_seq,self.docs_vocab,maxlen)

class Tensor_Sequence(DatasetJson):
    def __init__(self, filename, text_field, response_field, limit = 25, maxlen = 10, max_features = 5000):
        # quick setup
        self.filename = filename
        self.text_field = text_field
        self.response_field = response_field
        self.maxlen = maxlen

        # read json filename ()
        self.read_json(limit)
        # split response from data
        self.extract_data()

        # set up containers
        X_doc_seq = []
        Y_doc_seq = []

        # extract words for docs and sentences
        for idx, response in enumerate(self.response):

            doc = self.text[idx]

            # split into words and sentences
            x_words  = unicode(doc).split(" ")

            # each document is an observation (this is a traditional TDM format)
            if len(x_words) > 0:

                X_doc_seq.append(np.array(x_words))
                Y_doc_seq.append(response)

        # create response
        self.Y_doc_seq        = Y_doc_seq

        # create an index for the response (this will make it easier to create a response matrix later)
        self.docs_label_index = {label :idx for idx,label in enumerate(set(Y_doc_seq),start=0) }

        # create vocab for both points of view
        self.docs_vocab = self.create_vocab(X_doc_seq)

        # create sequence matrix for data
        self.X_doc_seq  = self.hot_vector_sequences(X_doc_seq,self.docs_vocab,maxlen)

#padding
def pad_vec_sequences(sequences,maxlen=40):
    new_sequences = []
    for sequence in sequences:
        orig_len, vec_len = np.shape(sequence) # can vec_len be 1?
        if orig_len < maxlen:
            new = np.zeros((maxlen,vec_len))
            new[maxlen-orig_len:,:] = sequence
        else:
            new = sequence[orig_len-maxlen:,:]
        new_sequences.append(new)
    return np.array(new_sequences)

def accuracy(model,x_test,y_test):
    predictions = model.predict(x_test)
    y_pred_vec = [ pred.argmax() for idx, pred in enumerate(predictions)]
    y_test_vec = [ pred.argmax() for idx, pred in enumerate(y_test)]
    return 100*float(np.array([ y_pred_vec[idx] == y_test_vec[idx]  for idx, pred in enumerate(predictions)]).sum())/len(y_pred_vec)

import numpy as np
def indicator_to_matrix(y,label_index):
    Y = np.zeros((len(y), len(label_index)))
    for i in range(len(y)):
        Y[i, label_index[y[i]]] = 1.
    return Y

#preprocessor
#from spacy.en import English
from os import listdir
from os.path import isfile, join
import numpy as np
import string
import json
import sys

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
                if sys.version_info[0] == 3: data.append(json.loads(line.decode("utf-8") )) # decode from bytes on Py3 :(
                else: data.append(json.loads(line))
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

    def W2V_vector_sequences(self,sequences,embeddings,dictionary,maxlen=40):
        new_sequences = []
        for sequence in sequences: # sequence here is a list of words
            orig_len = np.shape(sequence)
            orig_len = orig_len[0]
            # first create a numpy array of arrays
            new_seq = []
            for word in sequence:
                word_vec = Word_Vector(word,dictionary,embeddings)
                # word_vec[vocab[word.lower()]] = 1 # hot vector
                new_seq.append(word_vec)

            sequence = new_seq
            vec_len = len(word_vec)
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
        # this tdm is not weighted in any way # not true now using IDF
        vec_len = len(vocab)
        tdm = []
        n_docs = len(sequences)

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

        # convert to np array | over all sequences
        tdm = np.array(tdm)

        # now we can apply idf to the vector
        # tf*idf = frequency count * log_2 (total docs / docs containing word)
        # doc_vec has tf
        # total docs is len(sequences)
        # total word appearences is sum(tdm[:,word_index] > 0)

        import math

        word_doc_counts = np.sum(tdm>0,axis=0)

        idf = [math.log(i, 2) for i in (n_docs/word_doc_counts)]

        for idx, row in enumerate(tdm):
            tdm[idx] = row*idf # update with IDF


        # one could add IDF weighting here | done
        # this is actaully a document term matrix as its documents X terms but this is fine it will fit straight into a model
        return tdm

class DocumentTermMatrix(DatasetJson):
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

            # preprocess sentences
            doc = doc.replace("\n"," ")
            doc = doc.strip().lower()

            for c in string.punctuation: # terribly slow, but works
                doc = doc.replace(c,"")

            # split into words and sentences
            if sys.version_info[0] == 3: x_words  = doc.split(" ") # decode from bytes on Py3 :(
            else: x_words  = unicode(doc).split(" ")

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

            # preprocess sentences
            doc = doc.replace("\n"," ")
            doc = doc.strip().lower()

            for c in string.punctuation: # terribly slow, but works
                doc = doc.replace(c,"")

            if sys.version_info[0] == 3: x_words  = doc.split(" ") # decode from bytes on Py3 :(
            else: x_words  = unicode(doc).split(" ")

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
        self.X_idx_seq  = self.index_sequences(X_doc_seq,self.docs_vocab,maxlen)
        self.X_doc_seq  = X_doc_seq

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

            # preprocess sentences
            doc = doc.replace("\n"," ")
            doc = doc.strip().lower()

            for c in string.punctuation: # terribly slow, but works
                doc = doc.replace(c,"")

            # split into words and sentences
            if sys.version_info[0] == 3: x_words  = doc.decode("utf-8").split(" ") # decode from bytes on Py3
            else: x_words  = unicode(doc).split(" ")

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

class Tensor_Sequence_W2V(DatasetJson):
    def __init__(self, filename, text_field, response_field, embeddings, dictionary, limit = 25, maxlen = 10, max_features = 5000):
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

            # preprocess sentences
            doc = doc.replace("\n"," ")
            doc = doc.strip().lower()

            for c in string.punctuation: # terribly slow, but works
                doc = doc.replace(c,"")

            if sys.version_info[0] == 3: x_words  = doc.decode("utf-8").split(" ") # decode from bytes on Py3
            else: x_words  = unicode(doc).split(" ")

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
        self.X_doc_seq  = self.W2V_vector_sequences(X_doc_seq,embeddings,dictionary,maxlen)

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

def indicator_to_matrix(y,label_index):
    import numpy as np
    Y = np.zeros((len(y), len(label_index)))
    for i in range(len(y)):
        Y[i, label_index[y[i]]] = 1.
    return Y

def k_fold_cross_validation(X, K, randomise = False, seed = 123):
    """
    Generates K (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.

    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    """
    if randomise:
        import random
        from random import shuffle
        random.seed(seed) # for reproducibility
        X=list(X); shuffle(X)
    for k in xrange(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation

def Kfold_cv(X,K,random=True):
    result = []
    for training, validation in k_fold_cross_validation(X, K,random):
        result.append({"train": training, "test": validation})
    return result

# specific word vector
def Word_Vector(word,dictionary,embeddings):
    if word in dictionary: j = dictionary[word]
    else: j = dictionary["UNK"]
    return embeddings[j:j+1][0]

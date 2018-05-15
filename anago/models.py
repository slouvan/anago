import os
import tensorflow as tf
os.environ['PYTHONHASHSEED'] = '0'

import numpy as np
import random as rn

rn.seed(1234)
tf.set_random_seed(1234)

import keras.backend as K
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, Lambda
from keras.layers.merge import Concatenate
from keras.models import Model
# Not recommended; however this will correctly find the necessary contrib modules
from keras_contrib import *
#from anago.layers import ChainCRF
from keras_contrib.layers import CRF

from anago.layers import ChainCRF


class BaseModel(object):

    def __init__(self, config, embeddings, ntags):
        self.config = config
        self.embeddings = embeddings
        self.ntags = ntags
        self.model = None

    def predict(self, X, *args, **kwargs):
        y_pred = self.model.predict(X, batch_size=1)
        return y_pred

    def evaluate(self, X, y):
        score = self.model.evaluate(X, y, batch_size=1)
        return score

    def save(self, filepath):
        self.model.save_weights(filepath)

    def load(self, filepath):
        self.model.load_weights(filepath=filepath)

    def __getattr__(self, name):
        return getattr(self.model, name)


class SeqLabeling(BaseModel):
    """A Keras implementation of BiLSTM-CRF for sequence labeling.

    References
    --
    Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer.
    "Neural Architectures for Named Entity Recognition". Proceedings of NAACL 2016.
    https://arxiv.org/abs/1603.01360
    """

    def __init__(self, config, embeddings=None, ntags=None):
        # build word embedding

        word_ids = Input(batch_shape=(None, None), dtype='int32')
        self.config = config
        word_embeddings = None
        char_embeddings = None
        ner_embedings = None
        if embeddings is None:
            print("THE VOCABULARY SIZE 0 IS {}".format(config.vocab_size))
            word_embeddings = Embedding(input_dim=config.vocab_size,
                                        output_dim=config.word_embedding_size,
                                        mask_zero=True)(word_ids)
        else:
            print("THE VOCABULARY SIZE 1 IS {}".format(config.vocab_size))

            word_embeddings = Embedding(input_dim=embeddings.shape[0],
                                        output_dim=embeddings.shape[1],
                                        mask_zero=True,
                                        weights=[embeddings])(word_ids)

        if self.config.char_feature :
            # build character based word embedding
            char_ids = Input(batch_shape=(None, None, None), dtype='int32')
            #np.random.seed(1234)
            _embeddings = np.zeros([config.char_vocab_size, config.char_embedding_size])
            for idx in range(_embeddings.shape[0]):
                np.random.seed(1234)
                _embeddings[idx] = np.random.uniform(-0.25, 0.25, _embeddings.shape[1])

            char_embeddings = Embedding(input_dim=config.char_vocab_size,
                                    output_dim=config.char_embedding_size,
                                    mask_zero=True, weights=[_embeddings]
                                    )(char_ids)
            s = K.shape(char_embeddings)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], config.char_embedding_size)))(char_embeddings)

            fwd_state = LSTM(config.num_char_lstm_units, return_state=True)(char_embeddings)[-2]
            bwd_state = LSTM(config.num_char_lstm_units, return_state=True, go_backwards=True)(char_embeddings)[-2]
            char_embeddings = Concatenate(axis=-1)([fwd_state, bwd_state])
             # shape = (batch size, max sentence length, char hidden size)
            char_embeddings = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * config.num_char_lstm_units]))(char_embeddings)

            # combine characters and word
            if not self.config.ner_feature :
                x = Concatenate(axis=-1)([word_embeddings, char_embeddings])
            else :
                ner_ids = Input(batch_shape=(None, None), dtype='int32')
                _ner_embeddings = np.zeros([config.ner_vocab_size, config.ner_embedding_size])
                for idx in range(_ner_embeddings.shape[0]):
                    np.random.seed(1234)
                    _ner_embeddings[idx] = np.random.uniform(-0.25, 0.25, _ner_embeddings.shape[1])

                ner_embeddings = Embedding(input_dim=config.ner_vocab_size,
                                            output_dim=config.ner_embedding_size,
                                            mask_zero=True, weights=[_ner_embeddings]
                                            )(ner_ids)

                x = Concatenate(axis=-1)([word_embeddings, ner_embeddings, char_embeddings])

            x = Dropout(config.dropout)(x)
            x = Bidirectional(LSTM(units=config.num_word_lstm_units, return_sequences=True))(x)
        else :
            x = Bidirectional(LSTM(units=config.num_word_lstm_units, return_sequences=True))(word_embeddings)


        x = Dense(config.num_word_lstm_units, activation='relu')(x)
        x = Dense(ntags)(x)
        #self.crf = ChainCRF()
        self.crf = CRF(ntags)
        pred = self.crf(x)

        #sequence_lengths = Input(batch_shape=(None, 1), dtype='int32')
        if self.config.char_feature :
            if not self.config.ner_feature :
                self.model = Model(inputs=[word_ids, char_ids], outputs=[pred])
            else :
                self.model = Model(inputs=[word_ids, ner_ids,  char_ids], outputs=[pred])
        else :
            print("WITHOUT CHARACTER MODELS.PY")
            self.model = Model(inputs=word_ids, outputs=pred)


    def modify_model_for_transfer_learning(self):

        self.model.layers.pop()
        self.model.layers.pop()
        self.model.layers.pop()

        print("Chopping off the top one")
        print(self.model.summary())

        print("Modifying the top one")
        x = Dense(self.config.num_word_lstm_units, activation='relu')(self.model.layers[-1].output)
        print(self.ntags)
        x = Dense(self.ntags)(x)
        self.crf = CRF(self.ntags)
        pred = self.crf(x)
        self.model = Model(inputs=self.model.input, outputs=[pred])
        print(self.model.summary())

    def modify_model_for_transfer_learning_v2(self):
        print("VERSION 2 Guys")
        self.model.layers.pop()
        self.model.layers.pop()
        self.model.layers.pop()


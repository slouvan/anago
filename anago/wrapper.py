import os

import numpy as np
# Required, as usual
from keras.models import load_model

# Recommended method; requires knowledge of the underlying architecture of the model
from keras_contrib.layers.advanced_activations import PELU

# Not recommended; however this will correctly find the necessary contrib modules
from keras_contrib import *
from anago.config import ModelConfig, TrainingConfig
from anago.evaluator import Evaluator
from anago.models import SeqLabeling
from anago.preprocess import prepare_preprocessor, WordPreprocessor, filter_embeddings
from anago.tagger import Tagger
from anago.trainer import Trainer
import anago.config as cfg
from anago import utils
from anago.reader import load_data_and_labels, load_glove
import sys

class Sequence(object):

    config_file = 'config.json'
    weight_file = 'model_weights.h5'
    preprocessor_file = 'preprocessor.pkl'

    def __init__(self, char_emb_size=25, word_emb_size=100, char_lstm_units=25,
                 word_lstm_units=100, dropout=0.5, char_feature=True, crf=True,
                 batch_size=20, optimizer='adam', learning_rate=0.001, lr_decay=0.9,
                 clip_gradients=5.0, max_epoch=15, early_stopping=True, patience=3,
                 train_embeddings=True, max_checkpoints_to_keep=5, log_dir='',
                 embeddings=(), config_file = None, ner_feature = False):

        if config_file is not None:
            conf = self.load_config(config_file)
            self.model_config = ModelConfig(conf['char_emb_size'], conf['word_emb_size'], conf['char_lstm_units'],
                                            conf['word_lstm_units'], conf['dropout'], True if conf['char_feature'] == 1 else False, True if conf['crf'] == 1 else False, True if conf['ner_feature'] == 1 else False, conf['ner_emb_size'])
            self.training_config = TrainingConfig(conf['batch_size'], conf['optimizer'], conf['learning_rate'],
                                                  conf['lr_decay'], conf['clip_gradients'], conf['max_epoch'],
                                                  conf['early_stopping'], conf['patience'], conf['train_embeddings'],
                                                  conf['max_checkpoints_to_keep'])
            self.model = None
            self.p = None
            self.log_dir = conf['log_dir']
            self.char_feature = True if conf['char_feature'] == 1 else False
            self.ner_feature = True if conf['ner_feature'] == 1 else False

            if conf['use_pretrained_embedding'] == 1:
                self.embeddings = load_glove(conf['embeddings_file'])
            else:
                self.embeddings = embeddings

            self.conf = conf
        else:
            self.model_config = ModelConfig(char_emb_size, word_emb_size, char_lstm_units,
                                            word_lstm_units, dropout, char_feature, crf)
            self.training_config = TrainingConfig(batch_size, optimizer, learning_rate,
                                                  lr_decay, clip_gradients, max_epoch,
                                                  early_stopping, patience, train_embeddings,
                                                  max_checkpoints_to_keep)
            self.model = None
            self.p = None
            self.log_dir = log_dir
            self.embeddings = embeddings
            self.char_feature = char_feature

            self.conf = None

    def update_training_config(self, file_name):
        conf = self.load_config(file_name)

        self.training_config = TrainingConfig(conf['batch_size'], conf['optimizer'], conf['learning_rate'],
                                              conf['lr_decay'], conf['clip_gradients'], conf['max_epoch'],
                                              conf['early_stopping'], conf['patience'], conf['train_embeddings'],
                                              conf['max_checkpoints_to_keep'])
        self.log_dir = conf['log_dir']

    def load_config(self, file_name):
        conf = utils.load_config_file(file_name)
        return conf

    def train(self, x_train, y_train, x_valid=None, y_valid=None, vocab_init=None):
        self.p = prepare_preprocessor(x_train, y_train, use_char=self.char_feature, vocab_init=vocab_init, ner_feature = self.ner_feature)
        embeddings = filter_embeddings(self.embeddings, self.p.vocab_word,
                                       self.model_config.word_embedding_size)
        self.model_config.vocab_size = len(self.p.vocab_word)
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        if self.ner_feature :
            self.model_config.ner_vocab_size = len(self.p.vocab_ner_feature)

        self.model = SeqLabeling(self.model_config, embeddings, len(self.p.vocab_tag))
        trainer = Trainer(self.model,
                          self.training_config,
                          checkpoint_path=self.log_dir,
                          preprocessor=self.p)
        trainer.train(x_train, y_train, x_valid, y_valid)

    def modify_model_for_transfer_learning(self):
        self.model.modify_model_for_transfer_learning()

    def re_train(self,  x_train, y_train, x_valid=None, y_valid=None):
        self.p.update_vocab_tag(y_train)
        print(vars(self.training_config))
        print("Number of labels : {}".format(len(self.p.vocab_tag)))
        self.model.ntags = len(self.p.vocab_tag)
        self.model_config.vocab_size = len(self.p.vocab_word)
        self.model_config.char_vocab_size = len(self.p.vocab_char)
        print(type(self.model))
        self.model.modify_model_for_transfer_learning_v2()
        print(self.model.model.summary())
        print("LOG DIR : {}".format(self.log_dir))
        trainer = Trainer(self.model, self.training_config, checkpoint_path=self.log_dir, preprocessor=self.p)

        trainer.train(x_train, y_train, x_valid, y_valid)

    def eval(self, x_test, y_test, out_file_name=None):
        print("Inside eval {}".format(self.model))
        if self.model:
            evaluator = Evaluator(self.model, preprocessor=self.p)
            evaluator.eval(x_test, y_test, out_file_name=out_file_name)
        else:
            raise (OSError('Could not find a model. Call load(dir_path).'))

    def analyze(self, words):
        if self.model:
            tagger = Tagger(self.model, preprocessor=self.p)
            return tagger.analyze(words)
        else:
            raise (OSError('Could not find a model. Call load(dir_path).'))

    def analyze_sents(self, sents):
        if self.model :
            tagger = Tagger(self.model, preprocessor=self.p)
            return tagger.analyze_sents(sents)
        else:
            raise (OSError('Could not find a model. Call load(dir_path).'))

    def save(self, dir_path =''):
        if self.conf is not None:
            utils.clean_dir(self.conf['log_dir'])
            self.p.save(os.path.join(self.conf['log_dir'], self.preprocessor_file))
            self.model_config.vocab_size = len(self.p.vocab_word)
            self.model_config.char_vocab_size = len(self.p.vocab_char)
            self.model_config.save(os.path.join(self.conf['log_dir'], self.config_file))
            self.model.save(os.path.join(self.conf['log_dir'], self.weight_file))
        else:
            self.p.save(os.path.join(dir_path, self.preprocessor_file))
            self.model_config.vocab_size = len(self.p.vocab_word)
            self.model_config.char_vocab_size = len(self.p.vocab_char)
            self.model_config.save(os.path.join(dir_path, self.config_file))
            self.model.save(os.path.join(dir_path, self.weight_file))

    def save_config(self, dir_path =''):
        if self.conf is not None:
            self.p.save(os.path.join(self.conf['log_dir'], self.preprocessor_file))
            self.model_config.vocab_size = len(self.p.vocab_word)
            self.model_config.char_vocab_size = len(self.p.vocab_char)
            self.model_config.save(os.path.join(self.conf['log_dir'], self.config_file))
        else:
            self.p.save(os.path.join(dir_path, self.preprocessor_file))
            self.model_config.vocab_size = len(self.p.vocab_word)
            self.model_config.char_vocab_size = len(self.p.vocab_char)
            self.model_config.save(os.path.join(dir_path, self.config_file))

    @classmethod
    def load(cls, dir_path):
        self = cls()
        self.p = WordPreprocessor.load(os.path.join(dir_path, cls.preprocessor_file))
        print("PREPROCESSOR FILE : {}".format(os.path.join(dir_path, cls.preprocessor_file)))
        config = ModelConfig.load(os.path.join(dir_path, cls.config_file))
        dummy_embeddings = np.zeros((config.vocab_size, config.word_embedding_size), dtype=np.float32)
        self.model = SeqLabeling(config, dummy_embeddings, ntags=len(self.p.vocab_tag))
        self.model.load(filepath=os.path.join(dir_path, cls.weight_file))

        return self

    @classmethod
    def load_best_model(cls,  dir_path, model=None):
        self = cls()
        self.p = WordPreprocessor.load(os.path.join(dir_path, cls.preprocessor_file))
        print("PREPROCESSOR FILE : {}".format(os.path.join(dir_path, cls.preprocessor_file)))
        print("Aloha")

        config = ModelConfig.load(os.path.join(dir_path, cls.config_file))
        #print("CONFIG : {}".format(config.vocab_size))

        dummy_embeddings = np.zeros((config.vocab_size, config.word_embedding_size), dtype=np.float32)

        if model is None :
            self.model = SeqLabeling(config, dummy_embeddings, ntags=len(self.p.vocab_tag))
        else :
            self.model = model

        best_model_file_name = utils.get_best_model_file(dir_path)
        self.model.load(filepath=os.path.join(dir_path, best_model_file_name))
        print("Model : {}".format(self.model))
        return self
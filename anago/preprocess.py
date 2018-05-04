# -*- coding: utf-8 -*-
import itertools
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

UNK = '<UNK>'
PAD = '<PAD>'


class WordPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=False):

        self.lowercase = lowercase
        self.num_norm = num_norm
        self.char_feature = char_feature
        self.padding = padding
        self.return_lengths = return_lengths
        self.vocab_word = None
        self.vocab_char = None
        self.vocab_tag  = None
        self.vocab_init = vocab_init or {}

    def fit(self, X, y):
        words = {PAD: 0, UNK: 1}
        chars = {PAD: 0, UNK: 1}
        tags  = {PAD: 0, UNK: 1}
        print(self.lowercase)
        for w in set(itertools.chain(*X)) | set(self.vocab_init):
            w = self._lower(w)
            w = self._normalize_num(w)
            if w not in words:
                words[w] = len(words)

            if not self.char_feature:
                continue
            for c in w:
                if c not in chars:
                    chars[c] = len(chars)


        for t in itertools.chain(*y):
            if t not in tags:
                tags[t] = len(tags)
        #print("VOCAB BOSTON : {}".format(words['Boston']))

        self.vocab_word = words
        self.vocab_char = chars
        self.vocab_tag  = tags
        print("vocab char {}".format(self.vocab_char))
        print("vocab tag {}".format(self.vocab_tag))
        return self

    def transform(self, X, y=None):
        """transforms input(s)

        Args:
            X: list of list of words
            y: list of list of tags

        Returns:
            numpy array: sentences
            numpy array: tags

        Examples:
            >>> X = [['President', 'Obama', 'is', 'speaking']]
            >>> print(self.transform(X))
            [
                [
                    [1999, 1037, 22123, 48388],       # word ids
                ],
                [
                    [
                        [1, 2, 3, 4, 5, 6, 7, 8, 9],  # list of char ids
                        [1, 2, 3, 4, 5, 0, 0, 0, 0],  # 0 is a pad
                        [1, 2, 0, 0, 0, 0, 0, 0, 0],
                        [1, 2, 3, 4, 5, 6, 7, 8, 0]
                    ]
                ]
            ]
        """
        words = []
        chars = []
        lengths = []
        for sent in X:
            word_ids = []
            char_ids = []
            lengths.append(len(sent))
            for w in sent:
                if self.char_feature:
                    char_ids.append(self._get_char_ids(w))

                w = self._lower(w)
                w = self._normalize_num(w)
                if w in self.vocab_word:
                    word_id = self.vocab_word[w]
                else:
                    word_id = self.vocab_word[UNK]
                word_ids.append(word_id)

            words.append(word_ids)
            if self.char_feature:
                chars.append(char_ids)

        if y is not None:
            #y = [[self.vocab_tag[t] for t in sent] for sent in y]
            y = [[self.vocab_tag[t] if t in self.vocab_tag else self.vocab_tag[UNK] for t in sent] for sent in y]

        if self.padding:
            sents, y = self.pad_sequence(words, chars, y)
        else:
            sents = [words, chars]

        if self.return_lengths:
            #print("Di dalem return lengts {}".format(sents[0][0]))
            lengths = np.asarray(lengths, dtype=np.int32)
            #print("Di dalem return lengts LENGTH SHAPE {}".format(lengths.shape))
            lengths = lengths.reshape((lengths.shape[0], 1))
            #print("ISI LENGTH : {}".format(lengths))
            # dimension of length (jumlah_training_data, 1) isinya : [[15], [33]]
            sents.append(lengths)
        #print("Di luar return lengths {}".format(sents[0]))
        #print("X :{}".format(X[0]))
        #print(y.shape)

        return (sents, y) if y is not None else sents

    def inverse_transform(self, y):
        indice_tag = {i: t for t, i in self.vocab_tag.items()}
        return [indice_tag[y_] for y_ in y]

    def inverse_id_to_word(self, x):
        indice_word = {i: t for t, i in self.vocab_word.items()}
        return [indice_word[x_] for x_ in x]

    def _get_char_ids(self, word):
        return [self.vocab_char.get(c, self.vocab_char[UNK]) for c in word]

    def _lower(self, word):
        return word.lower() if self.lowercase else word

    def _normalize_num(self, word):
        if self.num_norm:
            return re.sub(r'[0-9０１２３４５６７８９]', r'0', word)
        else:
            return word

    def pad_sequence(self, word_ids, char_ids, labels=None):
        if labels:
            #print("LABELS 0 : {} length : {} ".format(labels, len(labels)))
            labels, _ = pad_sequences(labels, 0)
            labels = np.asarray(labels)
            #print("LABELS 1 : {} length : {} ".format(labels, len(labels)))
            labels = dense_to_one_hot(labels, len(self.vocab_tag), nlevels=2)
            #print("LABELS 2 : {} length : {} SHAPE : {}".format(labels, len(labels), labels.shape))
            # labels dimension (jumlah_training_data, max_panjang_kalimat, jumlah_label)

        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        word_ids = np.asarray(word_ids)
        #print("Njero pad_sequence {} ".format(word_ids[0])) # Numpy array [[1,2,3], [4,5,6]]
        #print("Type : {} ".format(type(word_ids[0]) ))  # [[1,2,3], [4,5,6]]
        #print("SHAPE 1 : {}".format(word_ids.shape))

        #print("LABEL : {}".format(labels[0]))
        if self.char_feature:
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            char_ids = np.asarray(char_ids)
            #print("SHAPE INPUT WORD_IDS: {}   CHAR_IDS: {}".format(word_ids.shape, char_ids.shape ))
            # dimensi word_ids : (jml_training_data, max_panjang_kalimat) char_ids : (jml_training_data, max_panjang_kalimat, max_panjang_kata)
            return [word_ids, char_ids], labels
        else:
            return word_ids, labels

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p


def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    if nlevels == 1:
        max_length = len(max(sequences, key=len))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        max_length_word = max(len(max(seq, key=len)) for seq in sequences)
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))

    return sequence_padded, sequence_length


def dense_to_one_hot(labels_dense, num_classes, nlevels=1):
    """Convert class labels from scalars to one-hot vectors."""
    if nlevels == 1:
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.int32)
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    elif nlevels == 2:
        # assume that labels_dense has same column length
        num_labels = labels_dense.shape[0]
        num_length = labels_dense.shape[1]
        labels_one_hot = np.zeros((num_labels, num_length, num_classes), dtype=np.int32)
        layer_idx = np.arange(num_labels).reshape(num_labels, 1)
        # this index selects each component separately
        component_idx = np.tile(np.arange(num_length), (num_labels, 1))
        # then we use `a` to select indices according to category label
        labels_one_hot[layer_idx, component_idx, labels_dense] = 1
        return labels_one_hot
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))


def prepare_preprocessor(X, y, use_char=True, vocab_init=None):
    p = WordPreprocessor(char_feature=use_char, vocab_init=vocab_init)
    p.fit(X, y)

    return p


def filter_embeddings(embeddings, vocab, dim):
    """Loads GloVe vectors in numpy array.

    Args:
        embeddings (dict): a dictionary of numpy array.
        vocab (dict): word_index lookup table.

    Returns:
        numpy array: an array of word embeddings.
    """
    _embeddings = np.zeros([len(vocab), dim])
    for idx in range(_embeddings.shape[0]):
        np.random.seed(1234)
        _embeddings[idx] = np.random.uniform(-0.25, 0.25, _embeddings.shape[1])

    for word in vocab:
        if word in embeddings:
            word_idx = vocab[word]
            _embeddings[word_idx] = embeddings[word]


    return _embeddings

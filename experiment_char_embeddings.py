import sys
sys.executable
import anago.config as cfg
'''
TRAINING_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/frame.train.conll"
DEV_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/frame.dev.conll"
TEST_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/frame.test.conll"
'''

'''
TRAINING_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/ner_conll2003_en.train.conll"
DEV_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/ner_conll2003_en.dev.conll"
TEST_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/ner_conll2003_en.test.conll"
'''

TRAINING_DATA_PATH = "/Users/slouvan/sandbox/anago/data/conll2003/en/ner/train.txt"
DEV_DATA_PATH = "/Users/slouvan/sandbox/anago/data/conll2003/en/ner/valid.txt"
TEST_DATA_PATH = "/Users/slouvan/sandbox/anago/data/conll2003/en/ner/test.txt"


'''
TRAINING_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/restaurant.train.conll"
DEV_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/restaurant.dev.conll"
TEST_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/restaurant.test.conll"
'''


'''
TRAINING_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/movie_MIT.train.conll"
DEV_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/movie_MIT.dev.conll"
TEST_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/movie_MIT.test.conll"
'''

'''
TRAINING_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/atis-2.train.conll"
DEV_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/atis-2.dev.conll"
TEST_DATA_PATH = "/Users/slouvan/sandbox/cross-domain/data/atis.test.iob.conll"
'''
import anago
from anago.reader import load_data_and_labels, load_glove


x_train, y_train = load_data_and_labels(TRAINING_DATA_PATH)
x_dev, y_dev = load_data_and_labels(DEV_DATA_PATH)
x_test, y_test = load_data_and_labels(TEST_DATA_PATH)

embeddings = load_glove("/Users/slouvan/sandbox/cross-domain/data/glove.6B.100d.txt")

model = anago.Sequence(embeddings=embeddings, char_feature=False)

model.train(x_train, y_train, x_dev, y_dev)
model.save(cfg.ANAGO_CHECKPOINT_DIR)
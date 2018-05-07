import sys
import pprint


print(sys.executable)
import argparse
import anago
from anago.reader import load_data_and_labels, load_glove
from anago.utils import load_config_file
from anago.utils import clean_dir
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Experiment Slot Filling")
    parser.add_argument("-c", "--config", dest="config_filename", help="Configuration file", metavar="FILE")

    args = parser.parse_args()

    cfg = load_config_file(args.config_filename)

    ner_feature = True if cfg['ner_feature'] == 1 else False

    x_train, y_train = load_data_and_labels(cfg['TRAINING_DATA_PATH'], ner_feature=ner_feature)
    x_dev, y_dev = load_data_and_labels(cfg['DEV_DATA_PATH'], ner_feature=ner_feature)
    x_test, y_test = load_data_and_labels(cfg['TEST_DATA_PATH'], ner_feature=ner_feature)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    clean_dir(cfg['log_dir'])

    model=anago.Sequence(config_file=args.config_filename)
    model.train(x_train, y_train, x_dev, y_dev)
    model.save_config(cfg['log_dir'])
    model.load_best_model(cfg['log_dir'])

    # load the best model that gives best performance in dev then evaluate on test
    model.eval(x_test, y_test)
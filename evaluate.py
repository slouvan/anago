import sys
import pprint


print(sys.executable)
import argparse
import anago
from anago.reader import load_data_and_labels, load_glove
from anago.utils import load_config_file
from anago.utils import clean_dir
import os
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Experiment Slot Filling")
    parser.add_argument("-c", "--config", dest="config_filename", help="Configuration file", metavar="FILE")

    args = parser.parse_args()

    cfg = load_config_file(args.config_filename)

    ner_feature = True if cfg['ner_feature'] == 1 else False
    x_dev, y_dev = load_data_and_labels(cfg['DEV_DATA_PATH'], ner_feature=ner_feature)
    x_test, y_test = load_data_and_labels(cfg['TEST_DATA_PATH'], ner_feature=ner_feature)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(cfg)
    #   clean_dir(cfg['log_dir'])

    model = anago.Sequence(config_file=args.config_filename)

    # load the best model that gives best performance in dev then evaluate on test
    model = model.load_best_model(cfg['log_dir'])

    print("Evaluation on DEV SET : ")
    model.eval(x_dev, y_dev, out_file_name=os.path.join(cfg['log_dir'],"dev_pred.txt"))

    print("Evaluation on TEST SET : ")
    model.eval(x_test, y_test, out_file_name=os.path.join(cfg['log_dir'],"test_pred.txt"))
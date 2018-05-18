import os
import sys
import anago.utils
import anago
import argparse
from anago.reader import load_data_and_labels
from anago.utils import  load_config_file
from anago.utils import  clean_dir

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Experiment Slot Filling")
    parser.add_argument("-c", "--config", dest="config_filename", help="Configuration file", metavar="FILE")
    parser.add_argument("-s", "--source", dest="source_task_model", help="Directory of the source task model", metavar="FILE")

    args = parser.parse_args()
    model = anago.Sequence()
    #model = model.load_best_model("/Users/slouvan/sandbox/cross-domain/models/anago/CONLL_2003_BASE_MODEL_RANDOM")
    model = model.load_best_model(args.source_task_model)

    cfg = load_config_file(args.config_filename)
    x_train, y_train = load_data_and_labels(cfg['TRAINING_DATA_PATH'])
    x_valid, y_valid = load_data_and_labels(cfg['DEV_DATA_PATH'])
    x_test , y_test = load_data_and_labels(cfg['TEST_DATA_PATH'])

    clean_dir(cfg['log_dir'])

    model.update_training_config(args.config_filename)
    model.re_train(x_train, y_train, x_valid, y_valid)
    model.save_config(cfg['log_dir'])

    best_model = model.load_best_model(cfg['log_dir'], model.model)
    print(best_model.model.model.summary())
    # load the best model that gives best performance in dev then evaluate on test

    model.eval(x_valid, y_valid, out_file_name=os.path.join(cfg['log_dir'], "dev_pred.txt"))
    model.eval(x_test, y_test, out_file_name=os.path.join(cfg['log_dir'], "test_pred.txt"))
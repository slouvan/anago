from anago.reader import batch_iter

from anago.metrics import F1score


class Evaluator(object):

    def __init__(self,
                 model,
                 preprocessor=None):

        self.model = model
        self.preprocessor = preprocessor

    def eval(self, x_test, y_test, out_file_name=None):

        # Prepare test data(steps, generator)
        train_steps, train_batches = batch_iter(x_test,
                                                y_test,
                                                batch_size=len(x_test),  # Todo: if batch_size=1, eval does not work.
                                                shuffle=False,
                                                preprocessor=self.preprocessor)

        # Build the evaluator and evaluate the model
        print("Type of x_test : {} len of x_test : {}".format(type(x_test), len(x_test)))
        print("X_TEST_[0] : {}".format(x_test[0]))

        f1score = F1score(train_steps, train_batches, self.preprocessor, mode="test", raw_data=x_test, out_file_name=out_file_name)
        f1score.model = self.model
        f1score.on_epoch_end(epoch=-1)  # epoch takes any integer.

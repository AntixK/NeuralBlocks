from collections import OrderedDict

class Logger:
    def __init__(self, metrics=['accuracy']):
        self.train_log = OrderedDict([
                                    ('Epoch',[]),
                                    ('Batch_idx',[]),
                                    ('Train_loss',[])])
        self.test_log = OrderedDict([
                                    ('Epoch',[]),
                                    ('Batch_idx',[]),
                                    ('Test_loss',[])])

        if not isinstance(metrics, list):
            raise ValueError("metrics must be a list.")

        for m in metrics:
            self.train_log['Train_'+m] = []
            self.test_log['Test_'+m] = []


    def add_log(self, values, is_train = True):

        if is_train:
            assert len(self.train_log) == len(values), \
                "Values list must (len={}) contain the same" \
                " number of elements as the train log (len={}).".format(len(self.train_log),len(values))

            for i, key in enumerate(self.train_log):
                self.train_log[key].append(values[i])
        else:
            assert len(self.test_log) == len(values), \
                "Values list must (len={}) contain the same" \
                " number of elements as the test log (len={}).".format(len(self.test_log), len(values))

            for i, key in enumerate(self.test_log):
                self.test_log[key].append(values[i])

    def get_logs(self):
        return {'Train_log':self.train_log,
                'Test_log':self.test_log}

    def get_epoch_log_cols(self):
        temp_list = []
        for key in self.train_log:
            if key != 'Batch_idx':
                temp_list.append(key)

        for key in self.test_log:
            if key not in ['Epoch', 'Batch_idx']:
                temp_list.append(key)

        return temp_list


    def get_epoch_log(self):
        temp_list = []
        for key, val in self.train_log.items():
            if key != 'Batch_idx':
                temp_list.append(val[-1])

        for key, val in self.test_log.items():
            if key not in ['Epoch', 'Batch_idx']:
                temp_list.append(val[-1])

        return temp_list
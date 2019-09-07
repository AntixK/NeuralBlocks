import os
import torch
import datetime
from time import time
from abc import ABC, abstractmethod
from fastprogress import  master_bar
from NeuralBlocks.trainers.logger import Logger


class Trainer(ABC):
    def __init__(self, use_cuda, metrics = None):
        """
        Trainer abstract class to handle some basic functions
        that all trainers require.

        :param use_cuda: Boolean to use the GPU for computation
        :param metrics: List of metrics to log
        """
        self.use_cuda = use_cuda
        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False
            raise Warning("CUDA not available! Using CPU instead.")

        if torch.cuda.is_available() and not self.use_cuda:
            raise Warning("CUDA is available on this machine. "
                          "Set use_cuda = True for faster computation.")
        self.log_handle = Logger(metrics=None, losses=None)

    @abstractmethod
    def train(self, epoch):
        pass

    @abstractmethod
    def test(self, epoch):
        pass

    def run(self, num_epochs, model_save_path = None):
        self.NUM_EPOCH = num_epochs
        self.mb = master_bar(range(self.NUM_EPOCH))

        if model_save_path is None:
            self.SAVE_PATH = os.getcwd()+'/Results/'
        else:
            self.SAVE_PATH = model_save_path

        if not os.path.isdir(self.SAVE_PATH):
            os.mkdir(self.SAVE_PATH)

        if not os.path.isdir(self.SAVE_PATH + 'checkpoint/'):
            os.mkdir(self.SAVE_PATH + 'checkpoint/')

        self.mb.write(self.log_handle.get_epoch_log_cols(), table=True)

        prev_time = time()
        for epoch in self.mb:
            self.train(epoch)
            self.test(epoch)
            self.mb.write(self.log_handle.get_epoch_log(), table=True)

        # Display Time taken
        wall_time = datetime.timedelta(seconds=(time() - prev_time))
        print("Wall Time: {}".format(wall_time))

    def get_logs(self):
        return self.log_handle.get_logs()

    def save_log(self, save_path, title="Experiment_log"):
        from numpy import save
        save(save_path+title+".npy", self.log_handle.get_logs())
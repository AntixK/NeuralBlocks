import torch
from NeuralBlocks.trainers import Trainer
import NeuralBlocks.trainers.metrics as M
from fastprogress import  progress_bar
from NeuralBlocks.trainers.logger import Logger

_metrics_ = ['accuracy', 'precision', 'recall', 'RMSE', 'MSE', 'MSE',
             'F1_score', 'top_3_accuracy', 'top_5_accuracy']


class SupervisedTrainer(Trainer):
    def __init__(self, model, data_bunch, optimizer, loss_function, metrics=None, use_cuda = True):
        """
        Trainer class for supervised DNN training - can be used for both Regression
        and Classification.

        :param model: model object derived from the nn.Module class
        :param data_bunch: List of train and test loaders
        :param optimizer: Optimiser object from torch.optim
        :param loss_function: Objective function to optimize
        :param metrics: Metrics to keep track during training.
                        Refer _metrics_ for the list of available metrics
        :param use_cuda: Boolean to use cuda if available
        """

        if metrics is None:
            metrics = ['accuracy']

        super(SupervisedTrainer, self).__init__(use_cuda, metrics)
        self.model = model
        self.trainloader = data_bunch[0]
        self.testloader = data_bunch[1]
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.criterion = loss_function
        self.CHECKPOINT_INTERVAL = 100

        if self.use_cuda:
            self.model = self.model.cuda()

        self.best_loss = float('Inf')
        self.metrics = metrics
        self.log_handle = Logger(metrics=metrics, losses=True)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(progress_bar(self.trainloader, parent=self.mb)):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Compute Loss
            train_loss += loss.item()
            train_loss = round(train_loss/(batch_idx + 1), 4)

            # Compute Metrics
            metric_results = []
            for metric in self.metrics:
                result = getattr(M, metric)(outputs, targets)
                metric_results.append(result)

            # Add to log
            self.log_handle.add_log([epoch + 1, batch_idx, train_loss]+metric_results, is_train=True)

            self.mb.child.comment = 'Train Loss:{:.3f}'.format(train_loss)

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(progress_bar(self.testloader, parent=self.mb)):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                # Compute Loss
                test_loss += loss.item()
                test_loss = round(test_loss/(batch_idx + 1), 4)

                # Compute metrics
                metric_results = []
                for metric in self.metrics:
                    result = getattr(M, metric)(outputs, targets)
                    metric_results.append(result)

                self.log_handle.add_log([epoch+1, batch_idx, test_loss]+metric_results, is_train=False)

                self.mb.child.comment = 'Test Loss:{:.3f}'.format(test_loss)

        # Save checkpoint.
        if test_loss > self.best_loss:
            # print('Saving..')
            state = {
                'model': self.model.state_dict(),
                'test loss': test_loss,
                'epoch': epoch,
            }
            torch.save(state, self.SAVE_PATH + 'checkpoint/ckpt.pth')
            self.best_loss = test_loss

        # Save model on keyboard interrupt
        # except KeyboardInterrupt:
        # torch.save(net.state_dict(), 'INTERRUPTED.pth')
        # print('Saved interrupt')
        # try:
        #     sys.exit(0)
        # except SystemExit:
        #     os._exit(0)

        # Have an option to export to ONNX format

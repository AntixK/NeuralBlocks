import torch
import os
from time import time
import datetime
from NeuralBlocks.trainers.logger import Logger
from fastprogress import master_bar, progress_bar

class SupervisedTrainer():
    def __init__(self, model, data_bunch, optimizer, loss_function, metrics=['accuracy'], use_cuda = True):

        self.model = model
        self.trainloader = data_bunch[0]
        self.testloader = data_bunch[1]
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.criterion = loss_function
        self.CHECKPOINT_INTERVAL = 100
        self.NUM_EPOCH = 0
        self.mb = master_bar(range(self.NUM_EPOCH))

        if self.use_cuda:
            self.model = self.model.cuda()

        if torch.cuda.is_available() and not self.use_cuda:
            raise Warning("CUDA is available on this machine. "
                                 "Set use_cuda = True for faster computation.")

        self.best_acc = 0
        self.log_handle = Logger(metrics=metrics)

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
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
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = round(100. * correct / total, 3)

            # Add to log
            self.log_handle.add_log([epoch + 1, batch_idx, train_loss, acc], is_train=True)

            self.mb.child.comment = 'Train Loss:{:.3f}'.format(train_loss)

            # if (batch_idx % self.CHECKPOINT_INTERVAL == 0):

                # print("Train Epoch [{:3d}/{:3d}]Batch [{:3d}/{:3d}] Loss: {:.3f} Acc {:.3f}%".format(epoch+1, self.NUM_EPOCH,
                #                                                                                      batch_idx,
                #                                                                                      len(self.trainloader),
                #                                                                                      train_loss / (
                #                                                                                                  batch_idx + 1),
                #                                                                                      100. * correct / total))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(progress_bar(self.testloader, parent=self.mb)):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                test_loss = round(test_loss/(batch_idx + 1), 4)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                acc = round(100. * correct / total, 3)

                self.log_handle.add_log([epoch+1, batch_idx, test_loss, acc], is_train=False)

                self.mb.child.comment = 'Test Loss:{:.3f}'.format(test_loss)

                # if (batch_idx % self.CHECKPOINT_INTERVAL == 0):
                #     print(
                #         "Test  Epoch [{:3d}/{:3d}]Batch [{:3d}/{:3d}] Loss: {:.3f} Acc {:.3f}%".format(epoch+1, self.NUM_EPOCH,
                #                                                                                       batch_idx,
                #                                                                                       len(self.testloader),
                #                                                                                       test_loss / (
                #                                                                                                   batch_idx + 1),
                #                                                                                       100. * correct / total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            # print('Saving..')
            state = {
                'model': self.model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, self.SAVE_PATH + 'checkpoint/ckpt.pth')
            self.best_acc = acc

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
        # Save model on keyboard interrupt
        # except KeyboardInterrupt:
        # torch.save(net.state_dict(), 'INTERRUPTED.pth')
        # print('Saved interrupt')
        # try:
        #     sys.exit(0)
        # except SystemExit:
        #     os._exit(0)

        # Have an option to export to ONNX format

    def get_logs(self):
        return  self.log_handle.get_logs()

    def save_log(self, save_path, title="Experiment_log"):
        from numpy import save
        save(save_path+title+".npy", self.log_handle.get_logs())



import torch

class SupervisedTrainer:
    def __init__(self, model, data_bunch, optimizer, loss_function, metrics, use_cuda = True):
        self.model = model
        self.trainloader = data_bunch[0]
        self.testloader = data_bunch[1]
        self.use_cuda = use_cuda
        self.optimizer = optimizer
        self.criterion = loss_function
        self.CHECKPOINT_INTERVAL = 100
        self.NUM_EPOCH = 0

        if self.use_cuda:
            self.model = self.model.cuda()

        self.best_acc = 0

        self.train_loss_log =[]
        self.train_acc_log = []
        self.test_loss_log =[]
        self.test_acc_log =[]

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            self.train_loss_log.append(train_loss / (batch_idx + 1))
            self.train_acc_log.append(100. * correct / total)

            if (batch_idx % self.CHECKPOINT_INTERVAL == 0):
                print("Train Epoch [{:3d}/{:3d}]Batch [{:3d}/{:3d}] Loss: {:.3f} Acc {:.3f}%".format(epoch, self.NUM_EPOCH,
                                                                                                     batch_idx,
                                                                                                     len(self.trainloader),
                                                                                                     train_loss / (
                                                                                                                 batch_idx + 1),
                                                                                                     100. * correct / total))

    def test(self, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                self.test_loss_log.append(test_loss / (batch_idx + 1))
                self.test_acc_log.append(100. * correct / total)

                if (batch_idx % self.CHECKPOINT_INTERVAL == 0):
                    print(
                        "Test Epoch [{:3d}/{:3d}]Batch [{:3d}/{:3d}] Loss: {:.3f} Acc {:.3f}%".format(epoch, self.NUM_EPOCH,
                                                                                                      batch_idx,
                                                                                                      len(self.testloader),
                                                                                                      test_loss / (
                                                                                                                  batch_idx + 1),
                                                                                                      100. * correct / total))

        # Save checkpoint.
        acc = 100. * correct / total
        if acc > self.best_acc:
            print('Saving..')
            state = {
                'model': self.model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            # if not os.path.isdir(SAVE_PATH + 'checkpoint'):
            #     os.mkdir(SAVE_PATH + 'checkpoint')
            # torch.save(state, SAVE_PATH + 'checkpoint/ckpt.pth')
            self.best_acc = acc

    def run(self, num_epochs):
        self.NUM_EPOCH = num_epochs
        from tqdm import tqdm_notebook
        for epoch in tqdm_notebook(range(self.NUM_EPOCH)):
            self.train(epoch)
            self.test(epoch)



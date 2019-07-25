import torch
import torchvision
import torchvision.transforms as transforms


PATH = '/home/robot/Anand/'
DATA_PATH = PATH+"NeuralBlocks/data_utils/datasets/EMNIST/"
SAVE_PATH = PATH+"NeuralBlocks/experiments/EMNIST/"
BATCH_SIZE = 128

transform_train = transforms.Compose(
    [#transforms.RandomAffine(degrees=7, translate=(0.1, 0.1), scale=(0.95, 1.05)),
     transforms.ToTensor(),
     transforms.Normalize((0.19036,), (0.34743,)),
    ])

transform_test = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.19036,), (0.34743,)),
    ])

trainset = torchvision.datasets.EMNIST(root=DATA_PATH, split='letters',train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

testset = torchvision.datasets.EMNIST(root=DATA_PATH, split='letters',train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
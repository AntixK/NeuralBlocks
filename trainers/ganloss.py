from abc import ABC, abstractmethod


class GANLoss(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def compute_loss(self, images, G_x, D_x, D_G_x):
        pass

    def __call__(self,images, G_x, D_x, D_G_x):
        return self.compute_loss(images, G_x, D_x, D_G_x)
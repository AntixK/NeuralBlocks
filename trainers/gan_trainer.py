import torch
from NeuralBlocks.trainers import Trainer
from fastprogress import progress_bar


class GANTrainer(Trainer):
    def __init__(self, G_model,
                       D_model,
                       dataloader,
                       gan_loss,
                       G_optimizer,
                       D_optimizer,
                       latent_dim,
                       use_cuda = True):
        super(GANTrainer, self).__init__(use_cuda)
        self.Generator = G_model
        self.Discriminator = D_model
        self.dataloader = dataloader
        self.Gan_loss = gan_loss
        self.z_dim = latent_dim
        self.G_optimizer = G_optimizer
        self.D_optimizer = D_optimizer

        if self.use_cuda:
            self.Generator.cuda()
            self.Discriminator.cuda()

    def train(self, epoch):
        """

        :param epoch:
        :return:
        """

        """
        Check if the dataloader has other variables other than the 
        image batch- like class labels. This is especially the case
        in standard image datasets for supervised learning.
        In user-defined dataloaders, this is usually not the case,
        as they only define a dataloader with the image batches.
        Here, we try to handle both in a suitable manner.
        """


        for i, (real_images, _) in enumerate(progress_bar(self.dataloader, parent=self.mb)):

            self.G_optimizer.zero_grad()
            z = torch.randn(real_images.shape[0], self.z_dim)

            if self.use_cuda:
                real_images = real_images.cuda()
                z = z.cuda()

            G_x = self.Generator(z)
            D_G_x = self.Discriminator(G_x)
            D_x = self.Discriminator(real_images)

            self.D_optimizer.zero_grad()

            # compute loss
            g_loss, d_loss = self.Gan_loss(images=real_images,
                                 G_x=G_x,
                                 D_x=D_x,
                                 D_G_x=D_G_x)
            g_loss.backward()
            self.G_optimizer.step()

            d_loss.backward()
            self.D_optimizer.step()

    def test(self, epoch):
        pass

    def generate_samples(self):
        pass


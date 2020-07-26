# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:38:15 2020

@authors: Andrey Prokpenko (e-mail: prokopenkoav@ornl.gov)
        : Debangshu Mukherjee (e-mail: mukherjeed@ornl.gov)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)
        : Vitaliy Starchenko  (e-mail: starchenkov@ornl.gov)
        : Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it) 

"""

import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pil
from torch.autograd import Variable

import CompetitiveGradientDescent as CGD


class CGANs_MLP_model(CGD.CGD.GANs_abstract_object.GANs_model):
    model_name = "C-GANs"

    def build_discriminator(self):
        D = CGD.CGD.ConditionalDiscriminator_MLP(self.data_dimension, self.n_classes)
        return D

    def build_generator(self, noise_dimension=100):
        self.noise_dimension = noise_dimension
        # n_out = numpy.prod(self.data_dimension)
        G = CGD.CGD.ConditionalGenerator_MLP(
            self.data_dimension, self.n_classes, self.noise_dimension
        )
        return G

    # loss = torch.nn.BCEWithLogitsLoss()
    # loss = binary_cross_entropy
    # loss = torch.nn.BCELoss()
    def train(
        self,
        loss=torch.nn.MSELoss(),
        lr_x=torch.tensor([0.001]),
        lr_y=torch.tensor([0.001]),
        optimizer_name="SGD",
        num_epochs=1,
        batch_size=100,
        verbose=True,
        save_path="./data_fake",
        label_smoothing=False,
        single_number=None,
        repeat_iterations=1,
    ):
        if single_number is not None:
            self.data = [i for i in self.data if i[1] == torch.tensor(single_number)]
            self.data_loader = torch.utils.data.DataLoader(
                self.data, batch_size=100, shuffle=True
            )
            self.num_test_samples = 5
            self.display_progress = 50
        else:
            self.data_loader = torch.utils.data.DataLoader(
                self.data, batch_size=100, shuffle=True
            )
            self.num_test_samples = 16
            self.display_progress = 100

        self.verbose = verbose
        self.save_path = save_path
        self.optimizer_initialize(loss, lr_x, lr_y, optimizer_name, self.n_classes)
        start = time.time()
        for e in range(num_epochs):
            self.print_verbose("######################################################")
            for n_batch, (real_batch, labels) in enumerate(self.data_loader):
                self.test_noise = CGD.CGD.noise(
                    self.num_test_samples, self.noise_dimension
                )
                # numpy.random.randint(0,10,self.num_test_samples)
                self.test_labels = Variable(
                    torch.LongTensor(
                        numpy.random.randint(0, self.n_classes, self.num_test_samples)
                    )
                )
                # self.test_labels = Variable(torch.LongTensor(np.random.randint(0, self.n_classes, batch_size)))
                N = real_batch.size(0)
                real_data = Variable(images_to_vectors(real_batch))
                labels = Variable(labels.type(torch.LongTensor))
                self.optimizer.G = self.G
                self.optimizer.D = self.D
                self.optimizer.zero_grad()

                if optimizer_name == "AdamCon":
                    error_real, error_fake, g_error = self.optimizer.step(
                        real_data, labels, N
                    )
                    self.D = self.optimizer.D
                    self.G = self.optimizer.G
                else:
                    raise RuntimeError("optimizer not supported, use AdamCon")

                self.D_error_real_history.append(error_real)
                self.D_error_fake_history.append(error_fake)
                self.G_error_history.append(g_error)

                self.print_verbose("Epoch: ", str(e + 1), "/", str(num_epochs))
                self.print_verbose("Batch Number: ", str(n_batch + 1))
                self.print_verbose(
                    "Error_discriminator__real: ",
                    "{:.5e}".format(error_real),
                    "Error_discriminator__fake: ",
                    "{:.5e}".format(error_fake),
                    "Error_generator: ",
                    "{:.5e}".format(g_error),
                )

                if (n_batch) % self.display_progress == 0:
                    test_images = CGD.CGD.vectors_to_images(
                        self.G(
                            self.test_noise.to(self.G.device),
                            self.test_labels.to(self.G.device),
                        ),
                        self.data_dimension,
                    )  # data_dimension: dimension of output image ex: [1,28,28]
                    self.save_images(e, n_batch, test_images)

            self.print_verbose("######################################################")
        end = time.time()
        self.print_verbose("Total Time[s]: ", str(end - start))

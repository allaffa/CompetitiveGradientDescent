# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 17:34:40 2020

@authors: Andrey Prokpenko (e-mail: prokopenkoav@ornl.gov)
        : Debangshu Mukherjee (e-mail: mukherjeed@ornl.gov)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)
        : Vitaliy Starchenko  (e-mail: starchenkov@ornl.gov)
        : Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it) 

"""
import torch
import numpy
import CompetitiveGradientDescent as CGD
import time
import PIL.Image as pil
import numpy as np
import os
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
from mpi4py import MPI


class GANs_model(metaclass=ABCMeta):
    def __init__(self, data, n_classes):
        self.mpi_comm_size = MPI.COMM_WORLD.Get_size()
        self.mpi_rank = MPI.COMM_WORLD.Get_rank()
        self.num_gpus = CGD.CGD.count_gpus()
        self.list_gpuIDs = CGD.CGD.get_gpus_list()
        self.data = data
        self.n_classes = n_classes
        self.data_dimension = self.data[0][0].numpy().shape
        self.D, self.G = self.build_models()
        self.D_error_real_history = []
        self.D_error_fake_history = []
        self.G_error_history = []

        if self.data_dimension[0] == 3:
            self.imtype = "RGB"
        else:
            self.imtype = "gray"

    @property
    @abstractmethod
    def model_name(self):
        pass

    def print_verbose(self, *args, **kwargs):
        if self.verbose and self.mpi_rank == 0:
            print(*args, **kwargs)

    def createFolder(self, directory):
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Creating directory. " + directory)

    def save_models(self):
        # G_directory = self.createFolder("/G_model")
        # D_directory = self.createFolder("/D_model")
        filename_D = "D_state_dict.pth"
        filename_G = "G_state_dict.pth"
        torch.save(self.G.state_dict(), filename_G)
        torch.save(self.D.state_dict(), filename_D)

    def build_models(self):
        self.discriminator_device = "cpu"
        self.generator_device = "cpu"

        D = self.build_discriminator()
        G = self.build_generator()

        # In peresence of GPUs available, map the models on the GPUs
        num_gpus = len(self.list_gpuIDs)
        if num_gpus > 0:
            rank = self.mpi_rank
            comm_size = self.mpi_comm_size

            num_ranks_with_2_gpus = max(
                min(num_gpus - comm_size, comm_size), 0
            )
            if rank < num_ranks_with_2_gpus:
                discriminator_gpu_index = 2 * rank + 0
                generator_gpu_index = discriminator_gpu_index + 1
            else:
                discriminator_gpu_index = (
                    rank % num_gpus + num_ranks_with_2_gpus
                )
                generator_gpu_index = discriminator_gpu_index

            self.discriminator_device = CGD.CGD.get_gpu(
                self.list_gpuIDs[discriminator_gpu_index]
            )
            self.generator_device = CGD.CGD.get_gpu(
                self.list_gpuIDs[generator_gpu_index]
            )

        D.to(self.discriminator_device)
        G.to(self.generator_device)

        return D, G

    @abstractmethod
    def build_discriminator(self):
        pass

    @abstractmethod
    def build_generator(self):
        pass

    def optimizer_initialize(
        self,
        loss,
        lr_x,
        lr_y,
        optimizer_name,
        n_classes,
        label_smoothing=False,
    ):
        if optimizer_name == "Jacobi":
            self.optimizer = Jacobi(
                self.G, self.D, loss, lr_x, lr_y, label_smoothing
            )
        elif optimizer_name == "CGD":
            self.optimizer = CGD.CGD.CGD(self.G, self.D, loss, lr_x)
        elif optimizer_name == "Newton":
            self.optimizer = CGD.CGD.Newton(self.G, self.D, loss, lr_x, lr_y)
        elif optimizer_name == "JacobiMultiCost":
            self.optimizer = CGD.CGD.JacobiMultiCost(
                self.G, self.D, loss, lr_x, lr_y
            )
        elif optimizer_name == "GaussSeidel":
            self.optimizer = CGD.CGD.GaussSeidel(
                self.G, self.D, loss, lr_x, lr_y
            )
        elif optimizer_name == "SGD":
            self.optimizer = CGD.CGD.SGD(self.G, self.D, loss, lr_x)
        elif optimizer_name == "Adam":
            self.optimizer = CGD.CGD.Adam(self.G, self.D, loss, lr_x, lr_y)
        elif optimizer_name == "CGD_multi":
            self.optimizer = CGD.CGD.CGDMultiCost(self.G, self.D, loss, lr_x)
        elif optimizer_name == "AdamCon":
            self.optimizer = CGD.CGD.AdamCon(
                self.G, self.D, loss, lr_x, lr_y, n_classes
            )
        else:
            raise RuntimeError("Optimizer type is not valid")

    def save_images(self, epoch_number, n_batch, images):
        count = 0
        for image_index in range(0, images.shape[0]):
            count = count + 1
            if self.imtype == "RGB":
                image = images[image_index]  # [0]
                image = image.detach().to("cpu").numpy()
                image = (image + 1) / 2
                image = image.transpose([1, 2, 0])
                self.createFolder(self.save_path)
                path = str(
                    self.save_path
                    + "/fake_image"
                    + "_MPI_rank_"
                    + str(self.mpi_rank)
                    + "_Epoch_"
                    + str(epoch_number + 1)
                    + "_Batch_"
                    + str(n_batch)
                    + "_N_image_"
                    + str(count)
                    + ".png"
                )
                plt.imsave(path, image)
            else:
                image = images[image_index][0]
                image = image.detach().to("cpu").numpy()
                image = (image + 1) / 2
                img = pil.fromarray(np.uint8(image * 255), "L")
                self.createFolder(self.save_path)
                path = str(
                    self.save_path
                    + "/fake_image"
                    + "_MPI_rank_"
                    + str(self.mpi_rank)
                    + "_Epoch_"
                    + str(epoch_number + 1)
                    + "_Batch_"
                    + str(n_batch)
                    + "_N_image_"
                    + str(count)
                    + ".png"
                )
                img.save(path)

    @abstractmethod
    def train(
        self,
        loss=torch.nn.BCEWithLogitsLoss(),
        lr_x=torch.tensor([0.001]),
        lr_y=torch.tensor([0.001]),
        optimizer_name="Jacobi",
        num_epochs=1,
        batch_size=100,
        verbose=True,
        save_path="./data_fake",
        label_smoothing=False,
        single_number=None,
        repeat_iterations=1,
    ):
        pass

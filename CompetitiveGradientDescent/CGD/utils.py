"""
@authors: Andrey Prokpenko (e-mail: prokopenkoav@ornl.gov)
        : Debangshu Mukherjee (e-mail: mukherjeed@ornl.gov)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)
        : Vitaliy Starchenko  (e-mail: starchenkov@ornl.gov)
        : Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it) 
"""
##########################################
import os
import numpy as np
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
from IPython import display
from matplotlib import pyplot as plt
import torch
from torch import autograd
from torch.autograd.variable import Variable
import math
from mpi4py import MPI

if torch.cuda.is_available():
    import pycuda
    from pycuda import compiler
    import pycuda.driver as drv

    drv.init()

################################################################


def ones_target(size):
    """
    Tensor containing ones, with shape = size
    """
    data = Variable(torch.ones(size, 1))
    return data


def ones_target_smooth(size):
    """
    Tensor containing 0.9s, with shape = size
    """
    data = torch.full((size,), 0.9)
    return data


def zeros_target(size):
    """
    Tensor containing zeros, with shape = size
    """
    data = Variable(torch.zeros(size, 1))
    return data


def zeros_target_smooth(size):
    """
    Tensor containing zeros, with shape = size
    """
    data = torch.full((size,), 0.1)

    return data


def noise(size, noise_size):
    """
    Generates a 1-d vector of gaussian sampled random values
    """
    n = Variable(torch.randn(size, noise_size))
    return n


def images_to_vectors(images):
    image_dim = images.size(1) * images.size(2) * images.size(3)
    return images.view(images.size(0), image_dim)


def vectors_to_images(vectors, array_dim):
    return vectors.view(
        vectors.size(0), array_dim[0], array_dim[1], array_dim[2]
    )


def images_to_vectors_cifar10(images):
    return images.view(images.size(0), 3072)


def vectors_to_images_cifar10(vectors):
    return vectors.view(vectors.size(0), 3, 32, 32)


def count_gpus():
    number = 0
    if torch.cuda.is_available():
        number = torch.cuda.device_count()
        print(number, " - GPUs found")
    else:
        print(number, " - GPU NOT found")
    return number


def get_gpus_list():
    gpu_list = []
    if torch.cuda.is_available():
        # print("%d device(s) found." % drv.Device.count())
        for ordinal in range(drv.Device.count()):
            dev = drv.Device(ordinal)
            # print (ordinal, dev.name())
            gpu_list.append(ordinal)
    return gpu_list


def get_gpu(number):
    gpus_list = get_gpus_list()
    if torch.cuda.is_available():
        if number not in gpus_list:
            raise ValueError(
                "The GPU ID:"
                + str(number)
                + " is not inside the list of GPUs available"
            )
        else:
            torch.cuda.device_count()
            device = torch.device(
                "cuda:" + str(number)
            )  # you can continue going on here, like cuda:1 cuda:2....etc.
            print(
                "MPI rank "
                + str(MPI.COMM_WORLD.Get_rank())
                + " running on the GPU with ID: "
                + str(number)
            )
    else:
        device = torch.device("cpu")
        print(
            "MPI rank "
            + str(MPI.COMM_WORLD.Get_rank())
            + " running on the CPU - GPU is NOT available"
        )
    return device


#############################################################################


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def Hvp_vec(grad_vec, params, vec, retain_graph=False):
    if torch.isnan(grad_vec).any():
        print("grad vec nan")
        raise ValueError("grad Nan")
    if torch.isnan(vec).any():
        print("vec nan")
        raise ValueError("vec Nan")
    try:
        grad_grad = autograd.grad(
            grad_vec, params, grad_outputs=vec, retain_graph=retain_graph
        )
        hvp = torch.cat([g.contiguous().view(-1) for g in grad_grad])
        if torch.isnan(hvp).any():
            print("hvp nan")
            raise ValueError("hvp Nan")
    except:
        # print('filling zero for None')
        grad_grad = autograd.grad(
            grad_vec,
            params,
            grad_outputs=vec,
            retain_graph=retain_graph,
            allow_unused=True,
        )
        grad_list = []
        for i, p in enumerate(params):
            if grad_grad[i] is None:
                grad_list.append(torch.zeros_like(p))
            else:
                grad_list.append(grad_grad[i].contiguous().view(-1))
        hvp = torch.cat(grad_list)
        if torch.isnan(hvp).any():
            raise ValueError("hvp Nan")
    return hvp


def hessian_vec(grad_vec, var, retain_graph=False):
    v = torch.ones_like(var)
    (vec,) = autograd.grad(
        grad_vec,
        var,
        grad_outputs=v,
        allow_unused=True,
        retain_graph=retain_graph,
    )
    return vec


def binary_cross_entropy(x, y):
    loss = -(x.log() * y + (1 - x).log() * (1 - y))
    return loss.mean()


class Richardson(object):
    def __init__(self, matrix, rhs, tol, maxiter, relaxation, verbose=False):
        """
        :param matrix: coefficient matrix
        :param rhs: right hand side
        :param tol: tolerance for stopping criterion based on the relative residual
        :param maxiter: maximum number of iterations
        :param relaxation: relaxation parameter for Richardson
        :param initial_guess: initial guess
        :return: matrix ** -1 * rhs
        """

        self.rhs = rhs
        self.matrix = matrix
        self.tol = tol
        self.maxiter = maxiter
        self.relaxation = relaxation
        self.rhs_norm = torch.norm(rhs, 2)
        self.iteration_count = 0
        self.verbose = verbose

    def print_verbose(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def solve(self, initial_guess):
        ## TODO: consider passing initial guess to solve()

        residual = self.rhs - self.matrix @ initial_guess
        residual_norm = residual.norm()
        relative_residual_norm = residual_norm / self.rhs_norm

        solution = initial_guess

        while (
            relative_residual_norm > self.tol
            and self.iteration_count < self.maxiter
        ):
            ## TODO: consider making all of these non-attributes and just return them
            solution = solution + self.relaxation * residual

            residual = self.rhs - torch.matmul(self.matrix, solution)
            residual_norm = residual.norm()
            relative_residual_norm = residual_norm / self.rhs_norm
            self.iteration_count += 1
            self.print_verbose(
                "Richardson converged in ",
                str(self.iteration_count),
                " iteration with relative residual norm: ",
                str(relative_residual_norm),
                end="...",
            )

        # Do not return because it's already an attribute
        return solution


def general_conjugate_gradient(
    grad_x,
    grad_y,
    x_params,
    y_params,
    kk,
    lr_x,
    lr_y,
    x=None,
    nsteps=10,
    residual_tol=1e-16,
    device_x=torch.device("cpu"),
    device_y=torch.device("cpu"),
):
    """

    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b:
    :param lr_x:
    :param lr_y:
    :param x:
    :param nsteps:
    :param residual_tol:
    :param device:
    :return: (I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) ** -1 * b

    """
    if x is None:
        x = torch.zeros(kk.shape[0], device=device_x)
    if grad_x.shape != kk.shape:
        raise RuntimeError("CG: hessian vector product shape mismatch")
    lr_x = lr_x.sqrt().to(device_x)
    lr_y = lr_y.to(device_y)

    mm = kk.clone().detach()
    mm = mm.to(device_x)
    jj = mm.clone().detach()
    jj = jj.to(device_x)
    rdotr = torch.dot(mm, mm)
    residual_tol = residual_tol * rdotr
    x_params = tuple(x_params)
    y_params = tuple(y_params)
    for i in range(nsteps):
        # To compute Avp
        # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True)
        h_1 = Hvp_vec(
            grad_vec=grad_x.to(device_x),
            params=y_params,
            vec=lr_x * jj,
            retain_graph=True,
        ).mul_(lr_y)
        # h_1.mul_(lr_y)
        # lr_y * D_yx * b
        # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=lr_y * h_1, retain_graph=True)
        h_2 = Hvp_vec(
            grad_vec=grad_y.to(device_x),
            params=x_params,
            vec=h_1.to(device_x),
            retain_graph=True,
        ).mul_(lr_x)
        # h_2.mul_(lr_x)
        # lr_x * D_xy * lr_y * D_yx * b
        Avp_ = jj + h_2

        alpha = rdotr / torch.dot(jj, Avp_)
        x.data.add_(alpha * jj)
        mm.data.add_(-alpha * Avp_)
        new_rdotr = torch.dot(mm, mm)
        beta = new_rdotr / rdotr
        jj = mm + beta * jj
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x, i + 1


#######################################################################
def general_conjugate_gradient_jacobi(
    grad_x,
    x_params,
    right_side,
    x=None,
    nsteps=10,
    residual_tol=1e-16,
    device=torch.device("cpu"),
):
    """

    :param grad_x:
    :param x_params:
    :param b:
    :param lr_x:
    :param x:
    :param nsteps:
    :param residual_tol:
    :param device:
    :return: (A) ** -1 * (right_side)

    """
    if x is None:
        x = torch.zeros(right_side.shape[0], device=device)
    else:
        x = x.to(device)

    right_side_clone1 = right_side.clone().detach()
    right_side_clone2 = right_side.clone().detach()
    right_side_clone1 = right_side_clone1.to(device)
    right_side_clone2 = right_side_clone2.to(device)

    rdotr = torch.dot(right_side_clone1, right_side_clone1)
    residual_tol = residual_tol * rdotr
    x_params = tuple(x_params)

    for i in range(nsteps):
        h_1 = Hvp_vec(
            grad_vec=grad_x.to(device),
            params=x_params,
            vec=2 * x,
            retain_graph=True,
        )
        H = -h_1.to(device) + x
        Avp_ = right_side_clone2 + H

        alpha = rdotr / torch.dot(right_side_clone2, Avp_)
        x.data.add_(alpha * right_side_clone2)
        right_side_clone1.data.add_(-alpha * Avp_)
        new_rdotr = torch.dot(right_side_clone1, right_side_clone1.to(device))
        beta = new_rdotr / rdotr
        right_side_clone2 = right_side_clone1 + beta * right_side_clone2
        rdotr = new_rdotr
        if rdotr < residual_tol:
            break
    return x, i + 1


###########################################

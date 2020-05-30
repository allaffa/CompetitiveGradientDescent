"""
@authors: Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)

"""
'''
CGD_shafer: Original implementation by Shafer
CGD: Ours implementation of CGD
Jacobi: Ours implementation of CGD with Jacobi method
JacobiMultiCost: Ours implementation of Jacobi with 2 different cost functions
GaussSeidel: GaussSeidel variation of the algorithm
Newton: Same algorithm but with pure essian term not set to identity
'''

import time
import torch
import numpy
from torch import Tensor
from torch import autograd
from torch.autograd import Variable
from torch.autograd import grad
from optimizers import *
from utils import *
from abc import ABCMeta, abstractmethod


class Optimizer(object, metaclass=ABCMeta):
    def __init__(self, G, D, criterion):
        self.count = 0
        self.criterion = criterion
        self.D = D
        self.G = G

    def zero_grad(self):
        zero_grad(self.G.parameters())
        zero_grad(self.D.parameters())

    @abstractmethod
    def step(self, real_data, N):
        pass


class CGD(Optimizer):
    def __init__(self, G, D, criterion, lr=1e-3):
        super(CGD, self).__init__(G, D, criterion)
        self.lr = lr

    def step(self, real_data, N):
        fake_data = self.G(noise(N, 100))
        prediction_real = self.D(real_data)
        error_real = self.criterion(prediction_real, ones_target(N))
        prediction_fake = self.D(fake_data)
        error_fake = self.criterion(prediction_fake, zeros_target(N))
        error_tot = error_fake + error_real
        errorG = self.criterion(prediction_fake, ones_target(N))
        grad_x = autograd.grad(
            error_tot,
            self.G.parameters(),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(
            error_tot,
            self.D.parameters(),
            create_graph=True,
            retain_graph=True,
        )
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
        scaled_grad_x = torch.mul(self.lr, grad_x_vec)
        scaled_grad_y = torch.mul(self.lr, grad_y_vec)
        # l = autograd.grad(grad_x_vec, discriminator.parameters(), grad_outputs = torch.ones_like(grad_x_vec))

        hvp_x_vec = Hvp_vec(
            grad_y_vec, self.G.parameters(), scaled_grad_y, retain_graph=True
        )  # D_xy * lr_y * grad_y
        hvp_y_vec = Hvp_vec(
            grad_x_vec, self.D.parameters(), scaled_grad_x, retain_graph=True
        )  # D_yx * lr_x * grad_x
        p_x = torch.add(
            grad_x_vec, -hvp_x_vec
        ).detach_()  # grad_x - D_xy * lr_y * grad_y
        p_y = torch.add(
            grad_y_vec, hvp_y_vec
        ).detach_()  # grad_y + D_yx * lr_x * grad_x
        p_x.mul_(self.lr.sqrt())
        cg_x, iter_num = general_conjugate_gradient(
            grad_x=grad_x_vec,
            grad_y=grad_y_vec,
            x_params=self.G.parameters(),
            y_params=self.D.parameters(),
            kk=p_x,
            x=None,
            nsteps=p_x.shape[0],
            lr_x=self.lr,
            lr_y=self.lr,
        )
        # cg_x.detach_().mul_(p_x_norm)
        # cg_x.detach_().mul_(p_x_norm)
        cg_x.detach_().mul_(self.lr.sqrt())  # delta x = lr_x.sqrt() * cg_x
        hcg = (
            Hvp_vec(grad_x_vec, self.D.parameters(), cg_x, retain_graph=True)
            .add_(grad_y_vec)
            .detach_()
        )
        # grad_y + D_yx * delta x
        cg_y = hcg.mul(-self.lr)

        return error_real.item(), error_fake.item(), errorG.item(), cg_x, cg_y


class CGD_shafer(Optimizer):
    def __init__(
        self, G, D, criterion, eps=1e-8, beta2=0.99, lr=1e-3, solve_x=False
    ):
        super(CGD_shafer, self).__init__(G, D, criterion)
        self.G_params = list(G.parameters())
        self.D_params = list(D.parameters())
        self.lr = lr
        self.square_avgx = None
        self.square_avgy = None
        self.beta2 = beta2
        self.eps = eps
        self.cg_x = None
        self.cg_y = None
        self.count = 0
        self.old_x = None
        self.old_y = None
        self.solve_x = solve_x

    def step(self, real_data, N):
        self.count += 1
        fake_data = self.G(
            noise(N, 100)
        )  # Second argument of noise is the noise_dimension parameter of build_generator
        d_pred_real = self.D(real_data)
        error_real = self.criterion(d_pred_real, ones_target(N))
        d_pred_fake = self.D(fake_data)
        error_fake = self.criterion(d_pred_fake, zeros_target(N))
        g_error = self.criterion(d_pred_fake, ones_target(N))
        loss = error_fake + error_real
        grad_x = autograd.grad(
            loss, self.G_params, create_graph=True, retain_graph=True
        )
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(
            loss, self.D_params, create_graph=True, retain_graph=True
        )
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        if self.square_avgx is None and self.square_avgy is None:
            self.square_avgx = torch.zeros(
                grad_x_vec.size(), requires_grad=False
            )
            self.square_avgy = torch.zeros(
                grad_y_vec.size(), requires_grad=False
            )
        self.square_avgx.mul_(self.beta2).addcmul_(
            1 - self.beta2, grad_x_vec.data, grad_x_vec.data
        )
        self.square_avgy.mul_(self.beta2).addcmul_(
            1 - self.beta2, grad_y_vec.data, grad_y_vec.data
        )

        # Initialization bias correction
        bias_correction2 = 1 - self.beta2 ** self.count

        lr_x = (
            math.sqrt(bias_correction2)
            * self.lr
            / self.square_avgx.sqrt().add(self.eps)
        )
        lr_y = (
            math.sqrt(bias_correction2)
            * self.lr
            / self.square_avgy.sqrt().add(self.eps)
        )

        scaled_grad_x = torch.mul(lr_x, grad_x_vec).detach()  # lr_x * grad_x
        scaled_grad_y = torch.mul(lr_y, grad_y_vec).detach()  # lr_y * grad_y

        hvp_x_vec = Hvp_vec(
            grad_y_vec, self.G_params, scaled_grad_y, retain_graph=True
        )  # D_xy * lr_y * grad_y
        hvp_y_vec = Hvp_vec(
            grad_x_vec, self.D_params, scaled_grad_x, retain_graph=True
        )  # D_yx * lr_x * grad_x

        p_x = torch.add(
            grad_x_vec, -hvp_x_vec
        ).detach_()  # grad_x - D_xy * lr_y * grad_y
        p_y = torch.add(
            grad_y_vec, hvp_y_vec
        ).detach_()  # grad_y + D_yx * lr_x * grad_x

        if self.solve_x:
            p_y.mul_(lr_y.sqrt())
            # p_y_norm = p_y.norm(p=2).detach_()
            # if self.old_y is not None:
            #     self.old_y = self.old_y / p_y_norm
            cg_y, self.iter_num = general_conjugate_gradient(
                grad_x=grad_y_vec,
                grad_y=grad_x_vec,
                x_params=self.D_params,
                y_params=self.G_params,
                kk=p_y,
                x=self.old_y,
                nsteps=p_y.shape[0] // 10000,
                lr_x=lr_y,
                lr_y=lr_x,
            )
            # cg_y.mul_(p_y_norm)
            cg_y.detach_().mul_(-lr_y.sqrt())
            hcg = (
                Hvp_vec(grad_y_vec, self.G_params, cg_y, retain_graph=True)
                .add_(grad_x_vec)
                .detach_()
            )
            # grad_x + D_xy * delta y
            cg_x = hcg.mul(lr_x)
            self.old_x = hcg.mul(lr_x.sqrt())
        else:

            p_x.mul_(lr_x.sqrt())
            # p_x_norm = p_x.norm(p=2).detach_()
            # if self.old_x is not None:
            #     self.old_x = self.old_x / p_x_norm
            cg_x, self.iter_num = general_conjugate_gradient(
                grad_x=grad_x_vec,
                grad_y=grad_y_vec,
                x_params=self.G_params,
                y_params=self.D_params,
                kk=p_x,
                x=self.old_x,
                nsteps=p_x.shape[0] // 10000,
                lr_x=lr_x,
                lr_y=lr_y,
            )
            # cg_x.detach_().mul_(p_x_norm)
            cg_x.detach_().mul_(lr_x.sqrt())  # delta x = lr_x.sqrt() * cg_x
            hcg = (
                Hvp_vec(grad_x_vec, self.D_params, cg_x, retain_graph=True)
                .add_(grad_y_vec)
                .detach_()
            )
            # grad_y + D_yx * delta x
            cg_y = hcg.mul(-lr_y)
            self.old_y = hcg.mul(lr_y.sqrt())

        return error_real.item(), error_fake.item(), g_error.item(), cg_x, cg_y


######################################


class Jacobi(Optimizer):
    def __init__(
        self, G, D, criterion, lr_x=1e-3, lr_y=1e-3, label_smoothing=False
    ):
        super(Jacobi, self).__init__(G, D, criterion)
        self.lr_x = lr_x
        self.lr_y = lr_y
        self.label_smoothing = label_smoothing

    def step(self, real_data, N):
        self.count += 1
        fake_data = self.G(
            noise(N, 100)
        )  # Second argument of noise is the noise_dimension parameter of build_generator
        # fake_data_copy = self.G(noise(N, 100))
        d_pred_real = self.D(real_data)
        # d_pred_real_copy = self.D(real_data)
        if self.label_smoothing:
            error_real = self.criterion(d_pred_real, ones_target_smooth(N))
        else:
            error_real = self.criterion(d_pred_real, ones_target(N))
        #    error_real_copy = self.criterion(d_pred_real_copy, ones_target(N) )
        d_pred_fake = self.D(fake_data)
        # d_pred_fake_copy = self.D(fake_data_copy)
        if self.label_smoothing:
            error_fake = self.criterion(d_pred_fake, zeros_target_smooth(N))
        else:
            error_fake = self.criterion(d_pred_fake, zeros_target(N))
        #    error_fake_copy = self.criterion(d_pred_fake_copy, zeros_target(N))
        g_error = self.criterion(d_pred_fake, ones_target(N))

        loss = error_fake + error_real
        # loss_copy = error_fake_copy + error_real_copy
        # loss = d_pred_real.mean() - d_pred_fake.mean()
        grad_x = autograd.grad(
            loss, self.G.parameters(), create_graph=True, retain_graph=True
        )
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(
            loss, self.D.parameters(), create_graph=True, retain_graph=True
        )
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        hvp_x_vec = Hvp_vec(
            grad_y_vec, self.G.parameters(), grad_y_vec, retain_graph=True
        )  # D_xy * grad_y
        hvp_y_vec = Hvp_vec(
            grad_x_vec, self.D.parameters(), grad_x_vec, retain_graph=False
        )  # D_yx * grad_x

        p_x = torch.add(
            grad_x_vec, 2 * hvp_x_vec
        ).detach_()  # grad_x +2 * D_xy * grad_y
        p_y = torch.add(
            -grad_y_vec, -2 * hvp_y_vec
        ).detach_()  # grad_y +2 * D_yx * grad_x
        p_x = p_x.mul_(self.lr_x)  # p_x.mul_(self.lr.sqrt())
        p_y = p_y.mul_(self.lr_y)  # p_y.mul_(self.lr.sqrt())

        return error_real.item(), error_fake.item(), g_error.item(), p_x, p_y


################################################
class GaussSeidel(Optimizer):
    def __init__(self, G, D, criterion, lr_x=1e-3, lr_y=1e-3):
        super(GaussSeidel, self).__init__(G, D, criterion)
        self.lr_x = lr_x
        self.lr_y = lr_y

    def step(self, real_data, N):
        fake_data = self.G(
            noise(N, 100)
        )  # Second argument of noise is the noise_dimension parameter of build_generator
        d_pred_real = self.D(real_data)
        error_real = self.criterion(d_pred_real, ones_target(N))
        d_pred_fake = self.D(fake_data)
        error_fake = self.criterion(d_pred_fake, zeros_target(N))
        g_error = self.criterion(d_pred_fake, ones_target(N))
        loss = error_fake + error_real
        # loss = d_pred_real.mean() - d_pred_fake.mean()
        grad_x = autograd.grad(
            loss, self.G.parameters(), create_graph=True, retain_graph=True
        )
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(
            loss, self.D.parameters(), create_graph=True, retain_graph=True
        )
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        hvp_x_vec = Hvp_vec(
            grad_y_vec, self.G.parameters(), grad_y_vec, retain_graph=True
        )  # D_xy * grad_y
        p_x = torch.add(
            grad_x_vec, 2 * hvp_x_vec
        ).detach_()  # grad_x + 2 * D_xy *  grad_y

        p_x.mul_(self.lr_x)

        index = 0
        for p in self.G.parameters():
            p.data.add_(p_x[index : index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != p_x.numel():
            raise RuntimeError('CG size mismatch')

        fake_data = self.G(
            noise(N, 100)
        )  # Second argument of noise is the noise_dimension parameter of build_generator
        d_pred_fake = self.D(fake_data)
        error_fake = self.criterion(d_pred_fake, zeros_target(N))
        g_error = self.criterion(d_pred_fake, ones_target(N))

        loss = error_fake + error_real

        grad_x = autograd.grad(
            loss, self.G.parameters(), create_graph=True, retain_graph=True
        )
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])

        hvp_y_vec = Hvp_vec(
            grad_x_vec, self.D.parameters(), grad_x_vec, retain_graph=True
        )  # D_yx * grad_x
        p_y = torch.add(
            -grad_y_vec, -2 * hvp_y_vec
        ).detach_()  # grad_y +2 * D_yx * x
        # p_x = torch.add(grad_x_vec, 2*hvp_x_vec).detach_()  # grad_x +2 * D_xy * y
        p_y.mul_(self.lr_y)

        index = 0
        for p in self.D.parameters():
            p.data.add_(p_y[index : index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != p_y.numel():
            raise RuntimeError('CG size mismatch')
        return error_real.item(), error_fake.item(), g_error.item()


##########################################


class SGD(Optimizer):
    def __init__(self, G, D, criterion, lr=1e-3):
        super(SGD, self).__init__(G, D, criterion)
        self.lr = lr

    def step(self, real_data, N):
        fake_data = self.G(
            noise(N, 100)
        )  # Second argument of noise is the noise_dimension parameter of build_generator
        d_pred_real = self.D(real_data)
        error_real = self.criterion(d_pred_real, ones_target(N))
        d_pred_fake = self.D(fake_data)
        error_fake = self.criterion(d_pred_fake, zeros_target(N))
        g_error = self.criterion(d_pred_fake, ones_target(N))
        loss = error_fake + error_real
        # loss = d_pred_real.mean() - d_pred_fake.mean()
        grad_x = autograd.grad(
            loss, self.G.parameters(), create_graph=True, retain_graph=True
        )
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(
            loss, self.D.parameters(), create_graph=True, retain_graph=True
        )
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
        scaled_grad_x = torch.mul(self.lr, grad_x_vec)
        scaled_grad_y = torch.mul(self.lr, grad_y_vec)

        p_x = scaled_grad_x
        p_y = scaled_grad_y

        return error_real.item(), error_fake.item(), g_error.item(), p_x, p_y


##############################################################################
class Newton(Optimizer):
    def __init__(self, G, D, criterion, lr_x=1e-3, lr_y=1e-3):
        super(Newton, self).__init__(G, D, criterion)
        self.lr_x = lr_x
        self.lr_y = lr_y

    def step(self, real_data, N):
        fake_data = self.G(
            noise(N, 100)
        )  # Second argument of noise is the noise_dimension parameter of build_generator
        d_pred_real = self.D(real_data)
        error_real = self.criterion(d_pred_real, ones_target(N))
        d_pred_fake = self.D(fake_data)
        error_fake = self.criterion(d_pred_fake, zeros_target(N))
        g_error = self.criterion(d_pred_fake, ones_target(N))
        loss = error_fake + error_real
        # loss = d_pred_real.mean() - d_pred_fake.mean()
        grad_x = autograd.grad(
            loss, self.G.parameters(), create_graph=True, retain_graph=True
        )
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(
            loss, self.D.parameters(), create_graph=True, retain_graph=True
        )
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])

        hvp_x_vec = Hvp_vec(
            grad_y_vec, self.G.parameters(), grad_y_vec, retain_graph=True
        )  # D_xy * grad_y
        hvp_y_vec = Hvp_vec(
            grad_x_vec, self.D.parameters(), grad_x_vec, retain_graph=True
        )  # D_yx * grad_x

        right_side_x = torch.add(
            grad_x_vec, 2 * hvp_x_vec
        ).detach_()  # grad_x + 2 * D_xy * grad_y
        right_side_y = torch.add(
            -grad_y_vec, -2 * hvp_y_vec
        ).detach_()  # grad_y + 2 * D_yx * grad_x

        p_x = general_conjugate_gradient_jacobi(
            grad_x_vec,
            self.G.parameters(),
            right_side_x,
            x=None,
            nsteps=1000,
            residual_tol=1e-16,
        )
        p_y = general_conjugate_gradient_jacobi(
            grad_y_vec,
            self.D.parameters(),
            right_side_y,
            x=None,
            nsteps=1000,
            residual_tol=1e-16,
        )
        p_x = p_x[0]
        p_y = p_y[0]

        p_x.mul_(self.lr_x)
        p_y.mul_(self.lr_y)

        return error_real.item(), error_fake.item(), g_error.item(), p_x, p_y


######################################################################################


class JacobiMultiCost(Optimizer):
    def __init__(self, G, D, criterion, lr_x=1e-3, lr_y=1e-3):
        super(JacobiMultiCost, self).__init__(G, D, criterion)
        self.lr_x = lr_x
        self.lr_y = lr_y

    def step(self, real_data, N):
        fake_data = self.G(
            noise(N, 100)
        )  # Second argument of noise is the noise_dimension parameter of build_generator
        d_pred_real = self.D(real_data)
        error_real = self.criterion(d_pred_real, ones_target(N))
        d_pred_fake = self.D(fake_data)
        error_fake = self.criterion(d_pred_fake, zeros_target(N))
        g_error = self.criterion(d_pred_fake, ones_target(N))

        f = error_fake + error_real  # f cost relative to discriminator
        g = g_error  # g cost relative to generator
        # loss = d_pred_real.mean() - d_pred_fake.mean()
        grad_f_x = autograd.grad(
            f, self.G.parameters(), create_graph=True, retain_graph=True
        )
        grad_g_x = autograd.grad(
            g, self.G.parameters(), create_graph=True, retain_graph=True
        )
        grad_f_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_f_x])
        grad_g_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_g_x])

        grad_f_y = autograd.grad(
            f, self.D.parameters(), create_graph=True, retain_graph=True
        )
        grad_g_y = autograd.grad(
            g, self.D.parameters(), create_graph=True, retain_graph=True
        )
        grad_f_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_f_y])
        grad_g_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_g_y])

        D_f_xy = Hvp_vec(
            grad_f_y_vec, self.G.parameters(), grad_f_y_vec, retain_graph=True
        )
        D_g_yx = Hvp_vec(
            grad_g_x_vec, self.D.parameters(), grad_g_x_vec, retain_graph=True
        )

        p_x = torch.add(
            grad_f_x_vec, 2 * D_f_xy
        ).detach_()  # grad_x + 2*D_xy * grad_y
        p_y = torch.add(
            grad_g_y_vec, 2 * D_g_yx
        ).detach_()  # grad_y + 2*D_yx * grad_x
        p_x.mul_(self.lr_x)
        p_y.mul_(self.lr_y)

        return error_real.item(), error_fake.item(), g_error.item(), p_x, p_y


#################################################################################

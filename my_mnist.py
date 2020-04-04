import torch
from modelli import *
from foo import *
from Dataloader import mnist_data
from optimizers import *
import matplotlib.pyplot as plt
from torch import nn, optim, autograd


data = mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)

#Learning rate
lr = torch.tensor([0.001])
lr_x = lr
lr_y = lr

generator = myGeneratorMNIST() # we will refer to this as x
discriminator = myDiscriminatorMNIST() # we will refer to this as y

loss =  torch.nn.BCEWithLogitsLoss()



def train_CGD(real_data, fake_data):
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N) )
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_tot = error_fake + error_real
    errorG = loss(prediction_fake, ones_target(N))
    print("Real - Discrim. Error: ", round(error_real.data.item(),5), "Fake - Discrim. Error: ", round(error_fake.data.item(),5), "Generator Error: ", round(errorG.data.item(),5))
    grad_x = autograd.grad(error_tot, generator.parameters(), create_graph=True, retain_graph=True, allow_unused= True)
    grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
    grad_y = autograd.grad(error_tot, discriminator.parameters(), create_graph=True, retain_graph=True)
    grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
    scaled_grad_x = torch.mul(lr,grad_x_vec)
    scaled_grad_y = torch.mul(lr,grad_y_vec)
    #l = autograd.grad(grad_x_vec, discriminator.parameters(), grad_outputs = torch.ones_like(grad_x_vec))
    
    hvp_x_vec = Hvp_vec(grad_y_vec, generator.parameters(), scaled_grad_y,retain_graph=True)  # D_xy * lr_y * grad_y 
    hvp_y_vec = Hvp_vec(grad_x_vec, discriminator.parameters(), scaled_grad_x,retain_graph=True)  # D_yx * lr_x * grad_x
    p_x = torch.add(grad_x_vec, - hvp_x_vec).detach_()  # grad_x - D_xy * lr_y * grad_y
    p_y = torch.add(grad_y_vec, hvp_y_vec).detach_()  # grad_y + D_yx * lr_x * grad_x
    p_x.mul_(lr_x.sqrt())
    cg_x, iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                             x_params=generator.parameters(),
                                                             y_params=discriminator.parameters(), kk=p_x,
                                                             x=None,
                                                             nsteps=p_x.shape[0] // 10000,
                                                             lr_x=lr_x, lr_y=lr_y,
                                                             )
            # cg_x.detach_().mul_(p_x_norm)
    # cg_x.detach_().mul_(p_x_norm)
    cg_x.detach_().mul_(lr_x.sqrt())  # delta x = lr_x.sqrt() * cg_x
    hcg = Hvp_vec(grad_x_vec, discriminator.parameters(), cg_x, retain_graph=True).add_(
    grad_y_vec).detach_()
            # grad_y + D_yx * delta x
    cg_y = hcg.mul(- lr_y)
    
    return cg_x, cg_y


def train_CGDJacobi(real_data, fake_data):
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N) )
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_tot = error_fake + error_real
    errorG = loss(prediction_fake, ones_target(N))
    print("Real - Discrim. Error: ", round(error_real.data.item(),5), "Fake - Discrim. Error: ", round(error_fake.data.item(),5), "Generator Error: ", round(errorG.data.item(),5))
    grad_x = autograd.grad(error_tot, generator.parameters(), create_graph=True, retain_graph=True, allow_unused= True)
    grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
    grad_y = autograd.grad(error_tot, discriminator.parameters(), create_graph=True, retain_graph=True)
    grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
    scaled_grad_x = torch.mul(lr,grad_x_vec)
    scaled_grad_y = torch.mul(lr,grad_y_vec)
    #l = autograd.grad(grad_x_vec, discriminator.parameters(), grad_outputs = torch.ones_like(grad_x_vec))
    
    hvp_x_vec = Hvp_vec(grad_y_vec, generator.parameters(), torch.cat([param.view(-1) for param in discriminator.parameters()]),retain_graph=True)  # D_xy * lr_y * y 
    hvp_y_vec = Hvp_vec(grad_x_vec, discriminator.parameters(), torch.cat([param.view(-1) for param in generator.parameters()]),retain_graph=True)  # D_yx * lr_x * x

    p_x = torch.add(- grad_x_vec, - hvp_x_vec).detach_()  # grad_x + D_xy * lr_y * y
    p_y = torch.add(grad_y_vec, hvp_y_vec).detach_()  # grad_y + D_yx * lr_x * x
    p_x.mul_(lr_x.sqrt())
    p_y.mul_(lr_y.sqrt())
    
    return p_x, p_y

def train_SGD(real_data, fake_data):
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N) )
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_tot = error_fake + error_real
    errorG = loss(prediction_fake, ones_target(N))
    print("Real - Discrim. Error: ", round(error_real.data.item(),5), "Fake - Discrim. Error: ", round(error_fake.data.item(),5), "Generator Error: ", round(errorG.data.item(),5))
    grad_x = autograd.grad(error_tot, generator.parameters(), create_graph=True, retain_graph=True, allow_unused= True)
    grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
    grad_y = autograd.grad(error_tot, discriminator.parameters(), create_graph=True, retain_graph=True)
    grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
    scaled_grad_x = torch.mul(lr,grad_x_vec)
    scaled_grad_y = torch.mul(lr,grad_y_vec)
    
    return scaled_grad_x, scaled_grad_y



num_test_samples = 16
test_noise = noise(num_test_samples)

num_epochs = 100
for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)
        real_data = Variable(images_to_vectors_mnist(real_batch))
        fake_data = generator(noise(N))
        #cg_x,cg_y  = train_CGD(real_data, fake_data)
        cg_x,cg_y  = train_CGDJacobi(real_data, fake_data)
        #cg_x,cg_y  = train_SGD(real_data, fake_data)

        index = 0
        for p in generator.parameters():
            p.data.add_(cg_x[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        if index != cg_x.numel():
            raise RuntimeError('CG size mismatch')
        index = 0
        for p in discriminator.parameters():
            p.data.add_(cg_y[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        
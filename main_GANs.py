#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Andrey Prokpenko (e-mail: prokopenkoav@ornl.gov)
        : Debangshu Mukherjee (e-mail: mukherjeed@ornl.gov)
        : Massimiliano Lupo Pasini (e-mail: lupopasinim@ornl.gov)
        : Nouamane Laanait (e-mail: laanaitn@ornl.gov)
        : Simona Perotto (e-mail: simona.perotto@polimi.it)
        : Vitaliy Starchenko  (e-mail: starchenkov@ornl.gov)
        : Vittorio Gabbi (e-mail: vittorio.gabbi@mail.polimi.it) 

Usage:
  main_GANS.py (-h | --help)
  main_GANS.py [-c CONFIG_FILE] [-m MODEL] [-e EPOCHS] [-o OPTIMIZER] [-r LEARNING_RATE] [-d DATASET] [--display] [--save] [--list]

Options:
  -h, --help                  Show this screen.
  --version                   Show version.
  --display                   Use matplotlib to plot results.
  --save                      Save the model.
  --list                      List available models.
  -c, --config=<str>          Filename containing configuration parameters
  -e, --epochs=<n>            Number of epochs [default: 100].
  -m, --model=<str>           Implementation of GANs model. Multi-layer perceptrons NN (MLP), convolutional NN (CNN), conditional MLP (C-GANs), conditional CNN (CNN-CGANs), resnet, (ResNet) [default: MLP], Wasserstein CNN (WGANS-CNN).
  -o, --optimizer=<str>       Optimizer name [default: Jacobi].
  -r, --learning_rate=<f>     Learning rate [default: 0.01].
  -d, --dataset=<srt>         Datased used for training. MNIST, CIFAR10, CIFAR100 [default: CIFAR10]
"""

from docopt import docopt
import matplotlib.pyplot as plt
import sys, os
import yaml
import torch
import mpi4py
import numpy as np

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

plt.rcParams.update({'font.size': 14})

from Dataloader import *

list_GANs = {}

models_dir = 'GANs_models'
# model classes must have identic name with python file in models directory
models_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), models_dir
)

# import GANs classes
for filename in os.listdir(models_path):
    modulename, ext = os.path.splitext(filename)
    if modulename != '__pycache__' and ext == '.py':
        subpackage = '{0}.{1}'.format(models_dir, modulename)
        obj = getattr(
            __import__(subpackage, globals(), locals(), [modulename]),
            modulename,
        )
        list_GANs.update({obj.model_name: obj})


MPI.Init()


def merge_args(cmdline_args, config_args):
    for key in config_args.keys():
        if key not in cmdline_args:
            sys.exit(
                'Error: unknown key in the configuration file \"{}\"'.format(
                    key
                )
            )

    args = {}
    args.update(cmdline_args)
    args.update(config_args)

    return args


def get_options():
    args = docopt(__doc__, version='Competitive Gradient Descent 0.0')

    # strip -- from names
    args = {key[2:]: value for key, value in args.items()}

    config_args = {}
    if args['config']:
        with open(args['config']) as f:
            config_args = yaml.load(f, Loader=yaml.FullLoader)

    # strip certain options
    # this serves 2 purposes:
    # - This option have already been parsed, and have no meaning as input
    #   parameters
    # - The config file options would be unallowed to contain them
    for skip_option in {'config', 'help'}:
        del args[skip_option]

    return merge_args(args, config_args)


if __name__ == '__main__':
    config = get_options()

    if config['list']:
        print("\n  Available models:")
        for key in list_GANs.keys():
            print("    {}".format(key))
        print("")
        exit(0)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print('-----------------')
        print('Input parameters:')
        print(yaml.dump(config))
        print('-----------------')

    epochs = int(config['epochs'])
    optimizer_name = config['optimizer']
    learning_rate = float(config['learning_rate'])
    model_name = config['model']

    if config['dataset'] == 'MNIST':
        data = mnist_data(rand_rotation=False, max_degree=90)
        n_classes = 10
    elif config['dataset'] == 'CIFAR10':
        data = cifar10_data()
        n_classes = 10
    elif config['dataset'] == 'CIFAR100':
        data = cifar100_data()
        n_classes = 100
    elif config['dataset'] == 'IMAGENET1K':
        data = imagenet_data()
        n_classes = 1000
    else:
        raise RuntimeError('Dataset not recognized')

    try:
        model = list_GANs[model_name](data, n_classes, model_name)
    except KeyError:
        sys.exit(
            '\n   *** Error. Specified model name: {} is not valid.\n'.format(
                model_name
            )
        )

    model.train(
        num_epochs=epochs,
        lr_x=torch.tensor([learning_rate]),
        lr_y=torch.tensor([learning_rate]),
        optimizer_name=optimizer_name,
        verbose=True,
        label_smoothing=False,
        single_number=None,
        repeat_iterations=1,
    )  # save_path = ''

    mpi_comm_size = MPI.COMM_WORLD.Get_size()
    mpi_rank = MPI.COMM_WORLD.Get_rank()

    if mpi_rank == 0:
        if config['save']:
            print("Saving the model...")
            model.save_models()

    if config['display']:
        plt.figure()
        plt.plot(
            [x for x in range(0, len(model.D_error_real_history))],
            model.D_error_real_history,
        )
        plt.plot(
            [x for x in range(0, len(model.D_error_fake_history))],
            model.D_error_fake_history,
        )
        plt.plot(
            [x for x in range(0, len(model.G_error_history))],
            model.G_error_history,
        )
        plt.xlabel('Epochs')
        plt.ylabel('Loss function value')
        plt.legend(
            [
                'Discriminator: Loss on Real Data',
                'Discriminator: Loss on Fake Data',
                'Generator: Loss',
            ]
        )
        plt.savefig('cost_report' + str(mpi_rank) + '.png')

        averageD_error_real_history = np.zeros(len(model.D_error_real_history))
        averageD_error_fake_history = np.zeros(len(model.D_error_fake_history))
        averageG_error_history = np.zeros(len(model.G_error_history))
        stdD_error_real_history = np.zeros(len(model.D_error_real_history))
        stdD_error_fake_history = np.zeros(len(model.D_error_fake_history))
        stdG_error_history = np.zeros(len(model.G_error_history))

        for i in range(0, len(model.D_error_real_history)):
            pointwise_error_LOC = np.zeros(mpi_comm_size)
            pointwise_error_LOC[mpi_rank] = model.D_error_real_history[i]
            pointwise_error_GLOB = np.zeros(mpi_comm_size)
            MPI.COMM_WORLD.Reduce(
                pointwise_error_LOC, pointwise_error_GLOB, op=MPI.SUM, root=0
            )
            mean_val = np.mean(pointwise_error_GLOB)
            averageD_error_real_history[i] = mean_val

        for i in range(0, len(model.D_error_fake_history)):
            pointwise_error_LOC = np.zeros(mpi_comm_size)
            pointwise_error_LOC[mpi_rank] = model.D_error_fake_history[i]
            pointwise_error_GLOB = np.zeros(mpi_comm_size)
            MPI.COMM_WORLD.Reduce(
                pointwise_error_LOC, pointwise_error_GLOB, op=MPI.SUM, root=0
            )
            mean_val = np.mean(pointwise_error_GLOB)
            averageD_error_fake_history[i] = mean_val

        for i in range(0, len(model.G_error_history)):
            pointwise_error_LOC = np.zeros(mpi_comm_size)
            pointwise_error_LOC[mpi_rank] = model.G_error_history[i]
            pointwise_error_GLOB = np.zeros(mpi_comm_size)
            MPI.COMM_WORLD.Reduce(
                pointwise_error_LOC, pointwise_error_GLOB, op=MPI.SUM, root=0
            )
            mean_val = np.mean(pointwise_error_GLOB)
            averageG_error_history[i] = mean_val

        for i in range(0, len(model.D_error_real_history)):
            pointwise_error_LOC = np.zeros(mpi_comm_size)
            pointwise_error_LOC[mpi_rank] = model.D_error_real_history[i]
            pointwise_error_GLOB = np.zeros(mpi_comm_size)
            MPI.COMM_WORLD.Reduce(
                pointwise_error_LOC, pointwise_error_GLOB, op=MPI.SUM, root=0
            )
            standard_deviation = np.std(pointwise_error_GLOB)
            stdD_error_real_history[i] = standard_deviation

        for i in range(0, len(model.D_error_fake_history)):
            pointwise_error_LOC = np.zeros(mpi_comm_size)
            pointwise_error_LOC[mpi_rank] = model.D_error_fake_history[i]
            pointwise_error_GLOB = np.zeros(mpi_comm_size)
            MPI.COMM_WORLD.Reduce(
                pointwise_error_LOC, pointwise_error_GLOB, op=MPI.SUM, root=0
            )
            standard_deviation = np.std(pointwise_error_GLOB)
            stdD_error_fake_history[i] = standard_deviation

        for i in range(0, len(model.G_error_history)):
            pointwise_error_LOC = np.zeros(mpi_comm_size)
            pointwise_error_LOC[mpi_rank] = model.G_error_history[i]
            pointwise_error_GLOB = np.zeros(mpi_comm_size)
            MPI.COMM_WORLD.Reduce(
                pointwise_error_LOC, pointwise_error_GLOB, op=MPI.SUM, root=0
            )
            standard_deviation = np.std(pointwise_error_GLOB)
            stdG_error_history[i] = standard_deviation

        if mpi_rank == 0:
            plt.figure()
            plt.plot(
                [x for x in range(0, len(averageD_error_real_history))],
                averageD_error_real_history,
                color='b',
            )
            ci = 1.96 * stdD_error_real_history
            plt.fill_between(
                [x for x in range(0, len(averageD_error_real_history))],
                (averageD_error_real_history - ci),
                (averageD_error_real_history + ci),
                color='b',
                alpha=0.2,
            )
            plt.plot(
                [x for x in range(0, len(averageD_error_fake_history))],
                averageD_error_fake_history,
                color='orange',
            )
            ci = 1.96 * stdD_error_fake_history
            plt.fill_between(
                [x for x in range(0, len(averageD_error_fake_history))],
                (averageD_error_fake_history - ci),
                (averageD_error_fake_history + ci),
                color='orange',
                alpha=0.2,
            )
            plt.plot(
                [x for x in range(0, len(averageG_error_history))],
                averageG_error_history,
                color='g',
            )
            ci = 1.96 * stdG_error_history
            plt.fill_between(
                [x for x in range(0, len(averageG_error_history))],
                (averageG_error_history - ci),
                (averageG_error_history + ci),
                color='g',
                alpha=0.2,
            )
            plt.xlabel('Epochs')
            plt.ylabel('Loss function value')
            plt.legend(
                [
                    'Discriminator: Loss on Real Data',
                    'Discriminator: Loss on Fake Data',
                    'Generator: Loss',
                ]
            )
            plt.savefig('average_cost_report.png')

MPI.Finalize()

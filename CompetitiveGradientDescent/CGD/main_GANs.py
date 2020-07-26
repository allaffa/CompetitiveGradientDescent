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
  main_GANS.py [-c CONFIG_FILE] [-m MODEL] [-e EPOCHS] [-o OPTIMIZER] [-r LEARNING_RATE] [--display] [--save] [--list]

Options:
  -h, --help                  Show this screen.
  --version                   Show version.
  --display                   Use matplotlib to plot results.
  --save                      Save the model.
  --list                      List available models.
  -c, --config=<str>          Filename containing configuration parameters
  -e, --epochs=<n>            Number of epochs [default: 100].
  -m, --model=<str>           Implementation of GANs model. Multi-layer perceptrons NN (MLP), convolutional NN (CNN) [default: MLP].
  -o, --optimizer=<str>       Optimizer name [default: Jacobi].
  -r, --learning_rate=<f>     Learning rate [default: 0.01].
"""

from docopt import docopt
import matplotlib.pyplot as plt
import sys, os
import yaml
import CompetitiveGradientDescent as CGD
import mpi4py

mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

# from Dataloader import *

list_GANs = {}

models_dir = "GANs_models"
# model classes must have identic name with python file in models directory
models_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), models_dir)

# import GANs classes
for filename in os.listdir(models_path):
    modulename, ext = os.path.splitext(filename)
    if modulename != "__pycache__" and ext == ".py":
        subpackage = "{0}.{1}".format(models_dir, modulename)
        obj = getattr(
            __import__(subpackage, globals(), locals(), [modulename]), modulename,
        )
        list_GANs.update({obj.model_name: obj})

"""
! READ ME !
Multi-layer perceptrons neural networks (MLP)
Convolutional neural networks (CNN)

X = Generator
Y = Discriminaror

Different learning rates for X and Y can only be used with 'Jacobi' and
'JacobiMultiCost' for the other optimizers the learning rate will be set to the
value of lr_x

Label smoothing variation is implemented only for optimizer 'Jacobi' and only
for GANs_object

Attribute save_models of both training object saves the state dicts of the
networks into 2 different folders inside your current directory
"""

MPI.Init()


def merge_args(cmdline_args, config_args):
    for key in config_args.keys():
        if key not in cmdline_args:
            sys.exit('Error: unknown key in the configuration file "{}"'.format(key))

    args = {}
    args.update(cmdline_args)
    args.update(config_args)

    return args


def get_options():
    args = docopt(__doc__, version="Competitive Gradient Descent 0.0")

    # strip -- from names
    args = {key[2:]: value for key, value in args.items()}

    config_args = {}
    if args["config"]:
        with open(args["config"]) as f:
            config_args = yaml.load(f, Loader=yaml.FullLoader)

    # strip certain options
    # this serves 2 purposes:
    # - This option have already been parsed, and have no meaning as input
    #   parameters
    # - The config file options would be unallowed to contain them
    for skip_option in {"config", "help"}:
        del args[skip_option]

    return merge_args(args, config_args)


if __name__ == "__main__":
    config = get_options()

    if config["list"]:
        print("\n  Available models:")
        for key in list_GANs.keys():
            print("    {}".format(key))
        print("")
        exit(0)

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("-----------------")
        print("Input parameters:")
        print(yaml.dump(config))
        print("-----------------")

    epochs = int(config["epochs"])
    optimizer_name = config["optimizer"]
    learning_rate = float(config["learning_rate"])

    model_name = config["model"]

    try:
        model = list_GANs[model_name](cifar10_data_dcgans(), 10)
    except KeyError:
        sys.exit(
            "\n   *** Error. Specified model name: {} is not valid.\n".format(
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

    if config["save"]:
        print("Saving the model...")
        model.save_models()

    if config["display"]:
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
            [x for x in range(0, len(model.G_error_history))], model.G_error_history,
        )
        plt.xlabel("Iterations")
        plt.ylabel("Loss function value")
        plt.legend(
            [
                "Discriminator: Loss on Real Data",
                "Discriminator: Loss on Fake Data",
                "Generator: Loss",
            ]
        )
        plt.savefig("cost_report.png")

MPI.Finalize()

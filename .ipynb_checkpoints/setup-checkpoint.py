from setuptools import setup, find_packages
import os

exec(open('CompetitiveGradientDescent/__version__.py').read())
with open('README.md') as f:
    long_description = f.read()

setup(name='CompetitiveGradientDescent',
      version=__version__,
      packages=find_packages(),
      url='https://github.com/allaffa/CompetitiveGradientDescent',
      license='MIT',
      zip_safe=False,
      install_requires=['torch >= 1.0.0',
                        'torchvision >= 0.2.2',
                        'numpy >= 1.13.0',
                        'matplotlib >= 2.2.0',
                        'pillow > 5.0.0',  
                        'mpi4py >= 3.0.0', 
                        'docopt >= 0.6.2', 
                        'tensorboardX >= 1.5'])
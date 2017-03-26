# Installation

## Plateform
The code runs on python3

## dependencies
In order to get the code running one need to have the following dependencies

### Install Pygame
`pip3 install hg+http://bitbucket.org/pygame/pygame`
you may be prompted to install additional dependencies to be able to launch the previous line

### Install Pymunk
The pymunk version on which all the physics of this project is based is based on pymunk version 4
`wget https://github.com/viblo/pymunk/archive/pymunk-4.0.0.tar.gz`
`tar zxvf pymunk-4.0.0.tar.gz`
`cd pymunk-pymukn-4.0.0/pymunk`
`2to3 -w *.py`
`cd ..` `python3 setup.py install`

# Running the code

## Training a model
On the `tester.py` file select the proper agents you want to train as well as the parameters in the main section
Then run `python3 tester.py`


## Playing pretrained model
`cd /Code`
`python3 tester.py`

Spiking neural network simulator which allows to simulate neuronal 
cultures placed over a Multi-Electrode Arrays (MEAs). 

MEAs can be used to stimulate and/or record the activity of the 
neuronal culture. Thus, these devices might enable the use of 
neuronal cultures as biological computing systems.
For example, it is possible to stimulate the neurons with input
patterns such as digits, record the resulting response and use it
to classify these patterns. Biological computation is attractive
because it has the potential to solve complex tasks with a very
limited energy consumption (as it happens, for example, in the brain).

Our goal is to explore whether it is possible to train neural cultures
on MEA devices to solve such kind of tasks. This simulator can be used
for finding an appropriate set of network parameters in order to solve
a given task. Once the appropriate parameters are known, it is then 
possible, through modern neurobiology techniques, to produce equivalent 
biological cultures.

A fundamental aspect of this work is to show the potential of the 
interplay between Computer Science and Neurobiology.

The simulator can be used as follows:  
In the file `configs.py` it is possible to specify custom experimental
configurations, defining the various parameters of the simulation. 
The script `runexp.py` can be used to launch an experiment session.
Experiments are performed on the MNIST dataset for digit recognition.  
The script `bioexp.py` is another experiment based on the following 
work on neuronal cultures: https://ieeexplore.ieee.org/document/1396377  
We simulated this experiment in order to obtain an estimate for the
parameters to be used as a starting point for successive simulations.   
Once you have set up your configuration, you can launch a training 
session with the command:  
`python runexp.py --config <config name> --mode train`  
Where `<config name>` is the name of one of the training configurations 
in the `config.py` file.  
To test the resulting model, type:  
`python runexp.py --config <config name> --mode test`  

REQUIREMENTS:  
- python 3.6
- matplotlib 3.1.1
- numpy 1.17.2
- scipy 1.4.1
- pytorch 1.4.0
- torchvision 0.5.0
- bindsnet 0.2.6


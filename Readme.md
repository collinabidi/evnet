# EvNet 2019
+ Author: Collin Abidi, David Langerman, Alex Johnson <<cba15@pitt.edu><david.langerman@nsf-shrec.org><amj92@pitt.edu>>
+ Group: P2 2019, Deep Learning


This program takes a user-specified Keras model and performs an evolutionary search to
find optimal CNN hyperparameters. The program generates mutated children models and runs 
accuracy tests, returning a more fit individual after a number of generations.

## Usage
1. Clone this repo on the host computer
2. Install the required Keras/Tensorflow dependencies
3. Start the program in the /src file with
'''
python main.py
'''
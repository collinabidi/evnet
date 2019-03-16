# based on https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html
import random
import json
import array

from deap import base
from deap import creator
from deap import tools

import pprint

population_size = 4
num_generations = 4
gene_length = 10

JSON_PATH = "/home/collin/Desktop/EA/"

with open(JSON_PATH+"vgg16.json", "r") as network_data:
    network = json.load(network_data)

layer_data = network["layers"]

# We want to maximize accuracy
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, typecode='i', fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Initialize with json file
def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)

def initIndividual(icls, content):
    return icls(content)

# Instantiate the initializers from json file
toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, JSON_PATH+"vgg16.json")

population = toolbox.population_guess()
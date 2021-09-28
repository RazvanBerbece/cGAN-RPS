#!/usr/bin/env python3

# Package Imports
import os
# from model.classes.prep.Dataset import Dataset
from model.classes.networks.generator.Generator import Generator

# dataset = Dataset(dataset='RockPaperScissors', seed=512, trainRatio='75%', batchSize=128)

generator = Generator(num_classes=3)
generator.activate_label()
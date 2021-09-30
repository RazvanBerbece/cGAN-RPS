#!/usr/bin/env python3

# Package Imports
import os
from model.classes.prep.Dataset import Dataset

dataset = Dataset(dataset='RockPaperScissors', seed=512, trainRatio='75%', batchSize=128)
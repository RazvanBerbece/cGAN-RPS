Tests ran on seed 345, on 60% train ratio and a batch size of 256.

# RESULTS OF TRAINING cGAN ON 15 EPOCHS WITH THE FOLLOWING PARAMETERISATION

| Hyperparam Test Run   |   #1   |   #2   |
| :---------            | :-:    | :-:    |
| D Embedding Size      | 5      |   ?    |
| G Embedding Size      | 5      |   ?    |
| D Initial Nodes       | 64     |   ?    |
| G Initial Nodes       | 64     |   ?    |
| Learning Rate D       | 0.01   |   ?    |
| Learning Rate G       | 0.001  |   ?    |
| D Optimiser           | Adamax |   ?    | 
| G Optimiser           | Adamax |   ?    |
| Add Noise             | True   |   ?    |
| Optimiser Beta Min    | 0.5    |   ?    |
| D Dropout Rate        | 0.4    |   ?    |
| Optimiser Beta Min    | 0.5    |   ?    |
| TRAINING TIME/EPOCH   | ~145s  |   ?    |
| TOTAL TRAINING TIME   | ~2175s |   ?    |
| AVG. LOSS D           | 57.47  |   ?    |
| AVG. LOSS D_G         | 8.16   |   ?    |
| AVG. LOSS G           | 141.80 |   ?    |

# Loss Plots & Last Generated Image

## Test Run 1
![Models Loss Plot](img/1/evaluation.png "Models Loss Plot")
![Last Generated Image](img/1/trainingSample.png "Last Generated Image")

## Test Run 2
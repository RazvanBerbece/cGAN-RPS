# cGAN-RPS
Conditional Generative Adversarial Network (cGAN) model for generating real 2D images of hands depicting Rock Paper Scissors shapes as used in the game.

Will use the CRISP-ML(Q) process model for the development of the project. 

# CRISP-ML(Q)

## CRISP-ML(Q) Progress
[x] Business and Data Understanding

[~] Data Engineering (Data Preparation)

[~] Machine Learning Model Engineering (~~Generator~~, ~~Discriminator~~, ~~Training Step~~, ~~Training~~, ~~ImageGen~~, Model Testing)

[ ] Quality Assurance for Machine Learning Applications

[~] Deployment (~~ImageGen API~~, Server, CI/CD)

[ ] Monitoring and Maintenance

## CRISP-ML(Q) Quality Assurance Flowchart
![CRISP-ML(Q) approach for quality assurance for each of the six phases](assets/crisp-ml-phase.jpeg "CRISP-ML(Q) approach for quality assurance for each of the six phases")

# Model Evaluation
The investigations are recorded & stored in ./docs/.
1. HYPERPARAM_<INTEGER> is a documentation of the test runs which aim to find the optimal hyperparameter values on a train run of <INTEGER> epochs. 
Each run documents :
    I. Hyperparameter values
    II. Time per epoch & Total time for whole process
    III. Model loss plot (D, G, D_G)
    IV. Last image generated by G on a batch

# Python

## Virtual Environment
The development process & testing should be carried in a virtual environment. When cloning the repository, there will be no virtual environment available, but it can be created & accessed & exit by running the terminal commands below (Unix/macOS) in the root of the project :
```console
python3 -m venv env
source env/bin/activate
deactivate
```
Trying to run 'python main.py' while not in the virtual environment won't work as the pip packages are installed in the virtual environment.

Libraries should be installed for the project while in the virtual environment.

## Data
The dataset comes from the default tensorflow_datasets package. It needs to be installed in the environment to be accessible. This can be done by running the command below.

```console
pip install tensorflow_datasets
```

## Libs
1. Tensorflow (& Keras)
2. sklearn
3. Other commonly-used libs (numpy, matplotlib, cv2, etc.)

Tensorflow and sklearn have to be installed using the commands below :
```console
pip install --upgrade tensorflow
pip install scikit-learn
```

# Resources
1. https://ml-ops.org/content/crisp-ml#:~:text=Overall%2C%20CRISP%2DML(Q,ensure%20the%20ML%20project's%20success
2. https://learnopencv.com/conditional-gan-cgan-in-pytorch-and-tensorflow/
3. https://developers.google.com/machine-learning/gan/generator
4. https://www.tensorflow.org/api_docs/python/tf/
5. https://stackoverflow.com/questions/
6. https://towardsdatascience.com/10-lessons-i-learned-training-generative-adversarial-networks-gans-for-a-year-c9071159628
7. https://towardsdatascience.com/beating-the-gan-game-afbcce0a20be
